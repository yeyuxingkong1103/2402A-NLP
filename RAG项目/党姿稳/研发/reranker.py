"""
reranker.py — BGE-rerank 重排序封装

混合检索召回的候选片段用交叉编码器逐条精算相关度，比向量余弦准得多。
优先使用 FlagEmbedding 的 FlagReranker；装不上时退回 sentence-transformers
的 CrossEncoder；两者都不可用时使用本地降级打分（余弦 + 关键词覆盖率），
保证链路在任何环境下都能跑通。

对外只暴露 rerank(query, documents, top_k)，重排结果写回 rerank_score 字段。
"""

from __future__ import annotations

import threading
from typing import Any

import config

_lock = threading.Lock()
_instance: "Reranker | None" = None


def _tokenize(text: str) -> set[str]:
    """分词成集合，用于降级打分；jieba 缺失时按字符切分。"""
    text = (text or "").lower()
    if not text:
        return set()
    try:
        import jieba

        tokens = {w.strip() for w in jieba.lcut(text) if w.strip()}
    except ImportError:
        tokens = set(text)
    return {t for t in tokens if len(t) > 1} or set(text)


class Reranker:
    """交叉编码器重排器，模型懒加载。"""

    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self.model_name = model_name
        self._model = None
        self._backend = "none"
        self._load_failed = False

    # ------------------------------------------------------------ 模型加载

    def _load(self) -> None:
        """按 FlagEmbedding -> CrossEncoder 的顺序尝试加载，失败则降级。"""
        if self._model is not None or self._load_failed:
            return

        with _lock:
            if self._model is not None or self._load_failed:
                return

            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_name, use_fp16=True)
                self._backend = "flagembedding"
                return
            except Exception:
                pass

            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                self._backend = "cross_encoder"
                return
            except Exception:
                self._load_failed = True
                self._backend = "fallback"

    @property
    def backend(self) -> str:
        self._load()
        return self._backend

    # ------------------------------------------------------------ 打分

    def _score_model(self, pairs: list[list[str]]) -> list[float]:
        """调用交叉编码器打分。"""
        if self._backend == "flagembedding":
            scores = self._model.compute_score(pairs, normalize=True)
        else:
            scores = self._model.predict(pairs)

        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]

    def _score_fallback(self, query: str, texts: list[str]) -> list[float]:
        """降级打分：向量余弦与关键词覆盖率的加权和。

        没有语义泛化能力，但能保证候选片段按实际相关程度大致排序。
        """
        import embeddings
        from rerank_filter import cosine_similarity

        query_tokens = _tokenize(query)
        query_vector = embeddings.encode_query(query)
        doc_vectors = embeddings.encode_texts(texts) if texts else []

        scores: list[float] = []
        for text, vector in zip(texts, doc_vectors):
            semantic = cosine_similarity(query_vector, vector)

            doc_tokens = _tokenize(text)
            overlap = len(query_tokens & doc_tokens) / len(query_tokens) if query_tokens else 0.0

            scores.append(round(0.5 * semantic + 0.5 * overlap, 6))
        return scores

    # ------------------------------------------------------------ 对外接口

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
        text_key: str = "text",
    ) -> list[dict]:
        """对候选文档重排，返回分数最高的 top_k 条。

        原始分数同时保留在 score 字段，重排分数写入 rerank_score。
        """
        if not documents:
            return []

        top_k = config.RERANK_TOP_K if top_k is None else top_k
        items = [dict(doc) for doc in documents]
        texts = [(doc.get(text_key) or "").strip() for doc in items]

        self._load()

        if self._backend in {"flagembedding", "cross_encoder"}:
            try:
                scores = self._score_model([[query, text] for text in texts])
            except Exception:
                # 推理阶段出错（显存不足等）也要退到降级方案，不能让请求挂掉
                self._backend = "fallback"
                self._load_failed = True
                scores = self._score_fallback(query, texts)
        else:
            scores = self._score_fallback(query, texts)

        for doc, score in zip(items, scores):
            doc["rerank_score"] = float(score)

        items.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return items[:top_k] if top_k > 0 else items


def get_reranker() -> Reranker:
    """返回全局重排器单例。"""
    global _instance
    if _instance is None:
        _instance = Reranker()
    return _instance


def rerank(
    query: str, documents: list[dict], top_k: int | None = None, text_key: str = "text"
) -> list[dict]:
    """便捷函数：用全局重排器对文档重排序。"""
    return get_reranker().rerank(query, documents, top_k=top_k, text_key=text_key)


def backend_name() -> str:
    """当前实际生效的重排后端，供日志和健康检查展示。"""
    return get_reranker().backend


if __name__ == "__main__":
    docs = [
        {"text": "劳动合同解除时，用人单位应当支付经济补偿金。"},
        {"text": "今天天气晴朗，适合外出散步。"},
        {"text": "劳动者提前三十日书面通知可以解除劳动合同。"},
    ]
    result = rerank("劳动合同怎么解除", docs, top_k=2)
    print(f"重排后端：{backend_name()}")
    for item in result:
        print(f"  {item['rerank_score']:.4f}  {item['text'][:20]}")
    assert "天气" not in result[0]["text"], "重排结果不合理"
    print("reranker 自检通过。")
