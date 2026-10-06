"""重排序（精排）与相似度阈值过滤。"""
from __future__ import annotations

from app.config import settings
from app.core.registry import get_reranker
from app.logging_conf import log


def rerank(query: str, docs: list[dict], top_k: int | None = None,
           threshold: float | None = None) -> list[dict]:
    """用 BGE-reranker 精排；模型不可用时回退按检索分数排序。"""
    if not docs:
        return []
    top_k = top_k or settings.rerank_top_k
    if not settings.enable_rerank:
        return docs[:top_k]

    reranker = get_reranker()
    if not reranker.available:
        log.warning("重排模型不可用，按检索分数回退排序")
        return sorted(docs, key=lambda d: -d.get("score", 0.0))[:top_k]

    try:
        scores = reranker.score(query, [d.get("text", "") for d in docs])
    except Exception as exc:  # noqa: BLE001
        log.warning("重排失败: %s", exc)
        return sorted(docs, key=lambda d: -d.get("score", 0.0))[:top_k]

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(docs, key=lambda d: -d["rerank_score"])
    if threshold is not None:
        filtered = [d for d in ranked if d["rerank_score"] >= threshold]
        ranked = filtered or ranked[:1]
    return ranked[:top_k]
