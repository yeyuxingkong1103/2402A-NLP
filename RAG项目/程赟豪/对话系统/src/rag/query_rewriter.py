"""Query 改写 / 扩写模块（RAG 优化）

文档.txt 要求 RAG 优化中包含「Query 改写，扩写」。这里实现：
  - rewrite(): 结合对话历史，把带指代的问题改写成独立完整的问题（共指消解）；
  - expand():  基于大模型生成多个召回变体，提高多路召回时的命中率。

大模型不可用时降级为轻量启发式（原文返回 / 停用词裁剪），保证链路永远可跑。
"""
import re
from typing import List, Optional, Dict, Any

from src.config import config
from src.utils.logger import logger

# 常见中文停用词（用于无 LLM 时的轻量扩写）
_STOPWORDS = {"的", "了", "呢", "吗", "啊", "呀", "吧", "么", "哦", "在", "是", "我",
              "你", "他", "她", "它", "我们", "你们", "他们", "这个", "那个", "什么",
              "怎么", "为什么", "请问", "一下", "可以", "能", "吗"}


class QueryRewriter:
    """查询改写器"""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.enabled = config.get('retrieval.query_rewrite', True)

    def rewrite(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """把当前问题改写成不依赖上下文也能独立检索的问题。"""
        if not self.enabled:
            return query
        if not history:
            return query

        if self.llm_client is not None:
            return self._rewrite_with_llm(query, history)

        return self._rewrite_heuristic(query, history)

    def expand(self, query: str, n: int = 2) -> List[str]:
        """扩写：返回 [原问题, 变体...]，用于多路召回。"""
        if not self.enabled:
            return [query]
        if self.llm_client is not None:
            return self._expand_with_llm(query, n)
        return [query] + self._expand_heuristic(query, n)

    # ---- LLM 路径 ----
    def _rewrite_with_llm(self, query: str, history: List[Dict[str, Any]]) -> str:
        try:
            hist_text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in history[-4:]
                if m.get('content')
            )
            resp = self.llm_client.chat(
                messages=[
                    {"role": "system", "content":
                        "你是查询改写助手。把用户的问题改写成不依赖对话历史也能理解的、"
                        "完整独立的检索问题，只输出改写后的问题本身，不要解释。"},
                    {"role": "user", "content": f"对话历史：\n{hist_text}\n\n当前问题：{query}\n\n改写后的问题："},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            text = (resp.get("content") or "").strip()
            if text:
                logger.info(f"Query 改写: {query[:30]} -> {text[:30]}")
                return text
        except Exception as e:
            logger.warning(f"Query 改写失败，回退原文: {e}")
        return query

    def _expand_with_llm(self, query: str, n: int) -> List[str]:
        try:
            resp = self.llm_client.chat(
                messages=[
                    {"role": "system", "content":
                        f"你是查询扩写助手。给出 {n} 个与用户问题语义等价但表达不同的检索查询，"
                        "每行一个，不要编号、不要解释。"},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            lines = [l.strip().lstrip('0123456789.- ').strip()
                     for l in (resp.get("content") or "").splitlines()]
            variants = [l for l in lines if l and l != query]
            out = [query] + variants[:n]
            return list(dict.fromkeys(out))
        except Exception as e:
            logger.warning(f"Query 扩写失败，回退原文: {e}")
            return [query]

    # ---- 无 LLM 的启发式 ----
    def _rewrite_heuristic(self, query: str, history: List[Dict[str, Any]]) -> str:
        # 简单共指消解：若问题以代词/指代开头，则把上一轮用户问题拼进来
        if re.match(r'^(它|他|她|这|那|其|该)', query):
            for m in reversed(history):
                if m.get('role') == 'user' and m.get('content'):
                    merged = f"{m['content']} {query}"
                    return merged
        return query

    def _expand_heuristic(self, query: str, n: int) -> List[str]:
        variants = []
        # 变体 1：去掉停用词后的关键词版
        stripped = "".join(ch for ch in query if ch not in _STOPWORDS)
        if stripped and stripped != query:
            variants.append(stripped)
        # 变体 2：仅保留较长词（≥2 字）的实体版
        words = [w for w in re.findall(r'[一-鿿]{2,}|[a-zA-Z0-9]{2,}', query)]
        if words:
            kw = " ".join(words)
            if kw and kw != query and kw not in variants:
                variants.append(kw)
        return variants[:n]
