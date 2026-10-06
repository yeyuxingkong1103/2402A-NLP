# -*- coding: utf-8 -*-
"""
重排器模块 - 3 种重排算法
工单编号: 人工智能 NLP-RAG-混合检索任务

1. TFIDFReranker: 基于 query-chunk TF-IDF 匹配分数
2. LLMReranker: 基于 LLM 的重排 (可选, 需 API Key)
3. FeedbackReranker: 基于用户反馈的自适应重排
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict

import config_v6 as config

logger = logging.getLogger(__name__)


class BaseReranker:
    """重排器基类"""
    name = "base"

    def rerank(self, query: str, candidates: List[Dict],
               query_tokens: List[str] = None) -> List[Dict]:
        """对 candidates 重排, 返回排序后的列表 (每个元素加 rerank_score)"""
        raise NotImplementedError


class TFIDFReranker(BaseReranker):
    """
    基于 TF-IDF 匹配的重排器
    计算 query 与每个 chunk 的词频匹配度, 调整原始分数
    """
    name = "tfidf"

    def __init__(self):
        try:
            import jieba
        except ImportError:
            jieba = None
        self.jieba = jieba

    def _tokenize(self, text: str) -> List[str]:
        if self.jieba:
            return [t.strip() for t in self.jieba.cut(text) if t.strip()]
        return [c for c in text if c.strip()]

    def rerank(self, query: str, candidates: List[Dict],
               query_tokens: List[str] = None) -> List[Dict]:
        q_tokens = query_tokens or self._tokenize(query)

        for c in candidates:
            text = c.get("text", "") or str(c.get("chunk", ""))
            # 计算 query token 在 chunk 中的覆盖率
            hits = sum(1 for t in q_tokens if t in text)
            coverage = hits / max(len(q_tokens), 1)
            # 与原始分数融合 (原始分 * 0.7 + 覆盖分 * 0.3)
            orig = c.get("score", 0)
            c["rerank_score"] = 0.7 * orig + 0.3 * coverage
            c["coverage"] = round(coverage, 4)
            c["hit_terms"] = [t for t in q_tokens if t in text]

        candidates.sort(key=lambda x: x.get("rerank_score", x.get("score", 0)),
                       reverse=True)
        logger.info(f"[Rerank TFIDF] {len(candidates)} 条, 覆盖 {candidates[0].get('coverage',0):.2f}")
        return candidates


class LLMReranker(BaseReranker):
    """
    基于 LLM 的重排器
    用 LLM 对 Top-K 候选打分排序
    """
    name = "llm"

    def __init__(self):
        self.available = bool(config.LLM_API_KEY)

    def rerank(self, query: str, candidates: List[Dict],
               query_tokens: List[str] = None) -> List[Dict]:
        if not self.available:
            logger.warning("[Rerank LLM] 未配置 LLM_API_KEY, 跳过重排")
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)
            return candidates

        # 只对 Top-10 做 LLM 重排 (节省 token)
        top10 = candidates[:10]
        try:
            scores = self._call_llm(query, top10)
            for c, s in zip(top10, scores):
                c["rerank_score"] = s
            # 其余候选保持原分数
            for c in candidates[10:]:
                c["rerank_score"] = c.get("score", 0)
        except Exception as e:
            logger.error(f"[Rerank LLM] 失败: {e}")
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)

        candidates.sort(key=lambda x: x.get("rerank_score", x.get("score", 0)),
                       reverse=True)
        return candidates

    def _call_llm(self, query: str, candidates: List[Dict]) -> List[float]:
        import requests
        prompt = (
            f"你是搜索结果重排器。请对以下 10 个文档片段根据与问题的相关性打分 (0-100)。\n"
            f"问题: {query}\n\n"
        )
        for i, c in enumerate(candidates):
            text = c.get("text", "")[:200]
            prompt += f"[{i+1}] {text}\n"
        prompt += "\n请按格式返回 10 个分数, 用逗号分隔 (如: 95,88,72,...):"

        messages = [
            {"role": "system", "content": "你是搜索结果重排器, 只返回 10 个数字, 用逗号分隔。"},
            {"role": "user", "content": prompt},
        ]
        url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers,
                             json={"model": config.LLM_MODEL,
                                   "messages": messages,
                                   "temperature": 0},
                             timeout=30)
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # 解析分数
        import re
        nums = re.findall(r"(\d+)", content)
        scores = [float(n) for n in nums[:len(candidates)]]
        while len(scores) < len(candidates):
            scores.append(50.0)
        return scores[:len(candidates)]


class FeedbackReranker(BaseReranker):
    """
    基于用户反馈的自适应重排器
    根据历史反馈 (👍👎) 学习哪些特征的 chunk 更受欢迎
    """
    name = "feedback"

    def __init__(self):
        self.feedback_file = config.FEEDBACK_FILE
        self._feedback_cache = self._load_feedback()

    def _load_feedback(self) -> Dict:
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"positive": [], "negative": [], "term_scores": {}}

    def _save_feedback(self):
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(self._feedback_cache, f, ensure_ascii=False, indent=2)

    def record_feedback(self, chunk_id: str, is_positive: bool):
        """记录用户反馈"""
        entry = {"chunk_id": chunk_id, "positive": is_positive}
        if is_positive:
            self._feedback_cache["positive"].append(entry)
        else:
            self._feedback_cache["negative"].append(entry)
        # 更新 term 权重
        if "term_scores" not in self._feedback_cache:
            self._feedback_cache["term_scores"] = {}
        self._save_feedback()

    def rerank(self, query: str, candidates: List[Dict],
               query_tokens: List[str] = None) -> List[Dict]:
        # 简单实现:
        # 1. 历史被点赞的 chunk_id → 加分
        # 2. 历史被点踩的 chunk_id → 减分
        pos_ids = {p["chunk_id"] for p in self._feedback_cache.get("positive", [])}
        neg_ids = {n["chunk_id"] for n in self._feedback_cache.get("negative", [])}

        for c in candidates:
            cid = str(c.get("id", c.get("chunk_id", "")))
            boost = 1.0
            if cid in pos_ids:
                boost = 1.3   # 点赞 → +30%
            elif cid in neg_ids:
                boost = 0.5   # 点踩 → -50%
            orig = c.get("score", 0)
            c["rerank_score"] = orig * boost
            c["feedback_boost"] = boost

        candidates.sort(key=lambda x: x.get("rerank_score", x.get("score", 0)),
                       reverse=True)
        return candidates


def get_reranker(name: str) -> BaseReranker:
    """获取重排器实例"""
    rerankers = {
        "tfidf": TFIDFReranker,
        "llm": LLMReranker,
        "feedback": FeedbackReranker,
    }
    cls = rerankers.get(name, TFIDFReranker)
    return cls()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = get_reranker("tfidf")
    print(f"重排器: {r.name}")
    candidates = [
        {"id": 1, "text": "武汉兴图新科注册资本7360万元", "score": 0.8},
        {"id": 2, "text": "武汉兴图新科法定代表人XXX", "score": 0.6},
    ]
    result = r.rerank("注册资本是多少", candidates)
    for c in result:
        print(f"  [{c['id']}] rerank={c.get('rerank_score',0):.3f} "
              f"coverage={c.get('coverage',0):.2f} hit={c.get('hit_terms',[])}")
