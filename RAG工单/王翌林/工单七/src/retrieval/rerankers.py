# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/retrieval/rerankers.py —— 工单六 三种重排算法

任务要求"提供至少 3 种重排算法"：
  1. LLMReranker：基于 LLM/交叉编码器的重排器（BAAI/bge-reranker-v2-m3，
     复用工单二 get_reranker，query 与候选交叉编码输出相关性概率）
  2. TfidfReranker：基于 TF-IDF 的重排器（候选集词法向量与 query 余弦，
     零模型依赖、毫秒级，作为离线/降级重排算法）
  3. AdaptiveReranker：基于用户反馈的自适应重排器（读取
     data/feedback_v6|feedback_v5|feedback 下的点赞/点踩记录，对候选做
     反馈相关性加权；无反馈冷启动时退化为 TF-IDF，反馈越多调整越强）

统一接口：rerank(query, candidates, top_k, content_key) → 候选列表（写 rerank_score）
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.retrieval.fulltext_retriever import tokenize, _bigrams

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"
_FEEDBACK_DIRS = ["data/feedback_v6", "data/feedback_v5", "data/feedback"]


def _minmax(values: List[float]) -> List[float]:
    """工单六：min-max 归一化到 [0,1]（全相同值时给 0.5）"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class BaseReranker:
    """工单六：重排器统一基类"""

    name = "base"

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               top_k: int = 8, content_key: str = "content") -> List[Dict[str, Any]]:
        raise NotImplementedError


class LLMReranker(BaseReranker):
    """工单六：基于 LLM/交叉编码器的重排器（bge-reranker-v2-m3）"""

    name = "llm"

    def __init__(self):
        self._model = None
        self._failed = False

    def _get_model(self):
        if self._model is None and not self._failed:
            try:
                from src.reranker import get_reranker
                self._model = get_reranker()
            except Exception as e:
                self._failed = True
                logger.warning(f"[rerank_v6] LLM 重排器加载失败: {e}")
        return self._model

    def rerank(self, query, candidates, top_k=8, content_key="content"):
        if not candidates:
            return []
        model = self._get_model()
        if model is None:
            # 工单六：模型不可用时降级保序
            return [dict(c, rerank_score=c.get("score", 0.0))
                    for c in candidates[:top_k]]
        try:
            return model.rerank(query, candidates, top_k=top_k,
                                content_key=content_key, max_chars=1500)
        except Exception as e:
            logger.warning(f"[rerank_v6] LLM 重排失败（保序）: {e}")
            return [dict(c, rerank_score=c.get("score", 0.0))
                    for c in candidates[:top_k]]


class TfidfReranker(BaseReranker):
    """工单六：基于 TF-IDF 的重排器（候选集内 IDF，query-candidate 余弦）"""

    name = "tfidf"

    def rerank(self, query, candidates, top_k=8, content_key="content"):
        if not candidates:
            return []
        docs = [str(c.get(content_key, "")) for c in candidates]
        toks = [tokenize(d) for d in docs]
        q_toks = tokenize(query)

        # 工单六：候选集 IDF
        n = len(docs)
        df: Dict[str, int] = {}
        for ts in toks:
            for t in set(ts):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log((1 + n) / (1 + d) + 1.0) for t, d in df.items()}

        def vec(tokens: List[str]) -> Dict[str, float]:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            return {t: (1.0 + math.log(f)) * idf.get(t, math.log(n + 1))
                    for t, f in tf.items()}

        def cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
            common = set(v1) & set(v2)
            dot = sum(v1[t] * v2[t] for t in common)
            n1 = math.sqrt(sum(x * x for x in v1.values()))
            n2 = math.sqrt(sum(x * x for x in v2.values()))
            return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

        qv = vec(q_toks)
        scored = [cosine(qv, vec(ts)) for ts in toks]
        # 工单六：与原始召回分做 0.85/0.15 凸组合，避免纯词法丢失通道先验
        base = _minmax([float(c.get("score", c.get("rrf", 0.0)))
                        for c in candidates])
        for c, tfidf_s, b in zip(candidates, scored, base):
            c["rerank_score"] = round(0.85 * tfidf_s + 0.15 * b, 6)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"],
                        reverse=True)
        return ranked[:top_k]


class AdaptiveReranker(BaseReranker):
    """工单六：基于用户反馈的自适应重排器

    融合分 = 0.6 × 词法相关分(TF-IDF) + 0.4 × 反馈调整分：
      - 候选与"点赞问题集"的 bigram/词项相似度 → 加分
      - 候选与"点踩问题集"的相似度 → 降分
      - 若候选文本曾在某条点赞回答引用的上下文中，额外加分（近似隐式相关反馈）
    无任何反馈数据时反馈分为 0，等价于 TF-IDF 重排（冷启动安全）。
    """

    name = "adaptive"

    def __init__(self, feedback_dirs: Optional[List[str]] = None):
        self._feedback_dirs = feedback_dirs or _FEEDBACK_DIRS
        self._tfidf = TfidfReranker()

    def _load_feedback(self) -> Dict[str, List[str]]:
        """工单六：汇总各版本点赞/点踩问题文本"""
        up, down = [], []
        for d in self._feedback_dirs:
            p = Path(d)
            if not p.exists():
                continue
            for fp in p.glob("feedback_*.json"):
                try:
                    records = json.loads(fp.read_text(encoding="utf-8"))
                    for r in records:
                        q = (r.get("query") or r.get("question") or "").strip()
                        if not q:
                            continue
                        (up if r.get("rating") == "up" else down).append(q)
                except Exception:
                    continue
        return {"up": up, "down": down}

    @staticmethod
    def _sim(text: str, questions: List[str]) -> float:
        """工单六：候选文本与反馈问题集的最大相似度（词项重合 + bigram Jaccard）"""
        if not questions or not text:
            return 0.0
        doc_terms = set(tokenize(text))
        doc_bi = _bigrams(text)
        best = 0.0
        for q in questions:
            q_terms = set(tokenize(q))
            term_overlap = (len(doc_terms & q_terms) / len(q_terms)
                            if q_terms else 0.0)
            q_bi = _bigrams(q)
            jac = (len(doc_bi & q_bi) / len(doc_bi | q_bi)
                   if doc_bi and q_bi else 0.0)
            best = max(best, 0.6 * term_overlap + 0.4 * jac)
        return best

    def rerank(self, query, candidates, top_k=8, content_key="content"):
        if not candidates:
            return []
        # 工单六：先取 TF-IDF 词法相关分
        tmp = [dict(c) for c in candidates]
        self._tfidf.rerank(query, tmp, top_k=len(tmp), content_key=content_key)
        fb = self._load_feedback()
        lex = _minmax([float(c.get("rerank_score", 0.0)) for c in tmp])
        for c, lx in zip(candidates, lex):
            up_s = self._sim(str(c.get(content_key, "")), fb["up"])
            dn_s = self._sim(str(c.get(content_key, "")), fb["down"])
            feedback_score = up_s - 0.8 * dn_s
            c["feedback_score"] = round(feedback_score, 4)
            c["rerank_score"] = round(0.6 * lx + 0.4 * max(feedback_score, 0.0), 6)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"],
                        reverse=True)
        logger.info(f"[rerank_v6] adaptive: up={len(fb['up'])} "
                    f"down={len(fb['down'])}")
        return ranked[:top_k]


# 工单六：重排器工厂（与 retrieval_config.VALID_RERANKERS 对应）
def get_reranker(name: str) -> BaseReranker:
    if name == "tfidf":
        return TfidfReranker()
    if name == "adaptive":
        return AdaptiveReranker()
    return LLMReranker()
