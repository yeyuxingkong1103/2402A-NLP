"""
rerank_filter.py — 余弦相似度计算、阈值过滤与去重

三处用到：
  1. 检索后按 SIMILARITY_THRESHOLD 丢弃低相干片段（检索质量兜底）
  2. 入库时按 DUPLICATE_THRESHOLD 去掉重复内容
  3. 混合检索里把两路分数归一化后加权融合，需要统一的相似度口径
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

# 短文本被长文本完整包含、且占比达到该值时，判定为同一内容的冗余副本
_CONTAINMENT_RATIO = 0.6


def _as_vector(value: Any) -> np.ndarray | None:
    """把 list / ndarray 统一成 float32 一维数组，非法输入返回 None。"""
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).ravel()
    except (ValueError, TypeError):
        return None
    return vector if vector.size else None


def cosine_similarity(a: Any, b: Any) -> float:
    """余弦相似度。向量已归一化时等价于点积，这里做通用实现以防万一。"""
    vec_a, vec_b = _as_vector(a), _as_vector(b)
    if vec_a is None or vec_b is None or vec_a.shape != vec_b.shape:
        return 0.0

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def normalize_scores(
    docs: Iterable[dict], score_key: str = "score", target_key: str = "norm_score"
) -> list[dict]:
    """把一组分数线性拉伸到 [0,1]，便于两路召回的结果加权融合。

    全部分数相同（含只有一条）时统一记为 1.0，避免除零。
    """
    items = [dict(doc) for doc in docs]
    if not items:
        return []

    scores = [float(doc.get(score_key, 0.0) or 0.0) for doc in items]
    low, high = min(scores), max(scores)
    span = high - low

    for doc, score in zip(items, scores):
        doc[target_key] = 1.0 if span <= 1e-9 else (score - low) / span
    return items


def filter_by_score(
    docs: Iterable[dict],
    threshold: float | None = None,
    score_key: str = "score",
) -> list[dict]:
    """按相似度阈值丢弃低相关片段，默认阈值取 config.SIMILARITY_THRESHOLD。"""
    import config

    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    return [dict(doc) for doc in docs if float(doc.get(score_key, 0.0) or 0.0) >= threshold]


def filter_by_similarity(
    query_vector: Any,
    docs: Iterable[dict],
    threshold: float | None = None,
    vector_key: str = "embedding",
) -> list[dict]:
    """用查询向量与文档向量重算余弦相似度再过滤。

    当检索返回的分数口径不一致（不同集合、不同 metric）时更可靠，
    结果会写回 score 字段。
    """
    import config

    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    query = _as_vector(query_vector)

    kept: list[dict] = []
    for doc in docs:
        item = dict(doc)
        if query is not None and item.get(vector_key) is not None:
            item["score"] = cosine_similarity(query, item[vector_key])
        if float(item.get("score", 0.0) or 0.0) >= threshold:
            kept.append(item)
    return kept


def _is_contained(text: str, other: str) -> bool:
    """短文本几乎完整地出现在长文本里，判定为同一内容的冗余副本。"""
    shorter, longer = (text, other) if len(text) <= len(other) else (other, text)
    if not longer:
        return False
    return shorter in longer and len(shorter) / len(longer) >= _CONTAINMENT_RATIO


def deduplicate(
    docs: Iterable[dict],
    threshold: float | None = None,
    vector_key: str = "embedding",
    text_key: str = "text",
) -> list[dict]:
    """内容去重：相似度超过阈值的只保留第一条。

    优先用向量比对；没有向量时退化为文本匹配。词汇型向量（hash 后端）对
    "原文后追加少量修饰"的近似重复不够敏感，故额外做包含关系判定。
    """
    import config

    threshold = config.DUPLICATE_THRESHOLD if threshold is None else threshold

    kept: list[dict] = []
    kept_vectors: list[np.ndarray] = []
    kept_texts: list[str] = []

    for doc in docs:
        item = dict(doc)
        text = (item.get(text_key) or "").strip()
        if not text:
            continue
        if any(_is_contained(text, other) for other in kept_texts):
            continue

        vector = _as_vector(item.get(vector_key))
        if vector is not None and any(
            cosine_similarity(vector, other) >= threshold for other in kept_vectors
        ):
            continue

        kept.append(item)
        kept_texts.append(text)
        if vector is not None:
            kept_vectors.append(vector)

    return kept


def top_k(docs: Iterable[dict], k: int, score_key: str = "score") -> list[dict]:
    """按分数降序取前 k 条。"""
    items = sorted(docs, key=lambda d: float(d.get(score_key, 0.0) or 0.0), reverse=True)
    return items[:k] if k > 0 else items


if __name__ == "__main__":
    import embeddings

    vecs = embeddings.encode_texts(["劳动合同如何解除", "劳动合同如何解除的说明", "今天天气真好"])
    docs = [
        {"text": "劳动合同如何解除", "embedding": vecs[0], "score": 0.9},
        {"text": "劳动合同如何解除的说明", "embedding": vecs[1], "score": 0.88},
        {"text": "今天天气真好", "embedding": vecs[2], "score": 0.1},
    ]

    unique = deduplicate(docs, threshold=0.95)
    print(f"去重后 {len(unique)} 条（原 3 条，前两条高度相似应合并）")
    assert len(unique) == 2, "去重结果不符合预期"

    kept = filter_by_score(docs, threshold=0.3)
    print(f"阈值过滤后 {len(kept)} 条")
    assert len(kept) == 2, "过滤结果不符合预期"

    sim = cosine_similarity(vecs[0], vecs[1])
    print(f"相似片段余弦 = {sim:.4f}")
    assert sim > 0.5, "相似文本的余弦值过低"
    print("rerank_filter 自检通过。")
