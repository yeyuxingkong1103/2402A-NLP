from __future__ import annotations

import numpy as np


def evaluate_retrieval(model, records: list[dict], documents: list[dict], top_k: int = 3, adapted: bool = False) -> dict:
    if not records:
        raise ValueError("评估集为空")
    if not documents:
        raise ValueError("文档集为空")
    if not 1 <= top_k <= len(documents):
        raise ValueError(f"Top K必须在1到{len(documents)}之间")

    identifiers = [str(item.get("chunk_id", "")) for item in documents]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("文档chunk_id必须唯一")
    id_to_index = {identifier: index for index, identifier in enumerate(identifiers)}
    for record in records:
        if str(record.get("positive_id", "")) not in id_to_index:
            raise ValueError(f"问题{record.get('question_id')}的正例不在候选文档中")

    query_vectors = model.encode([record["query"] for record in records], adapted=adapted)
    document_vectors = model.encode([item["text"] for item in documents], adapted=adapted)
    similarities = query_vectors @ document_vectors.T

    details: list[dict] = []
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for row_index, record in enumerate(records):
        order = np.argsort(-similarities[row_index], kind="stable")
        positive_index = id_to_index[str(record["positive_id"])]
        rank = int(np.where(order == positive_index)[0][0]) + 1
        positive_score = float(similarities[row_index, positive_index])
        negative_score = float(max(similarities[row_index, index] for index in range(len(documents)) if index != positive_index))
        reciprocal_ranks.append(1 / rank)
        recalls.append(float(rank <= top_k))
        positive_scores.append(positive_score)
        negative_scores.append(negative_score)
        details.append(
            {
                "question_id": record.get("question_id"),
                "query": record["query"],
                "positive_id": record["positive_id"],
                "rank": rank,
                "positive_similarity": round(positive_score, 4),
                "top_negative_similarity": round(negative_score, 4),
                "retrieved_ids": [identifiers[index] for index in order[:top_k]],
            }
        )

    mean_positive = float(np.mean(positive_scores))
    mean_negative = float(np.mean(negative_scores))
    return {
        "count": len(records),
        "top_k": top_k,
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "mean_positive_similarity": round(mean_positive, 4),
        "mean_top_negative_similarity": round(mean_negative, 4),
        "similarity_gap": round(mean_positive - mean_negative, 4),
        "queries": details,
    }


def compare_models(model, records: list[dict], documents: list[dict], top_k: int = 3) -> dict:
    before = evaluate_retrieval(model, records, documents, top_k=top_k, adapted=False)
    after = evaluate_retrieval(model, records, documents, top_k=top_k, adapted=True)
    metrics = ("recall_at_k", "mrr", "similarity_gap")
    return {
        "before": before,
        "after": after,
        "delta": {key: round(after[key] - before[key], 4) for key in metrics},
    }
