"""Evaluate a dataset with Ragas when installed, with a local fallback summary."""

from __future__ import annotations

from ragas_data import EvaluationRecord


def local_metrics(records: list[EvaluationRecord]) -> dict[str, float]:
    grounded = [bool(item.contexts) and bool(item.answer.strip()) for item in records]
    answer_match = [bool(item.ground_truth.strip()) and item.ground_truth[:12] in item.answer for item in records]
    return {
        "context_available_rate": sum(grounded) / max(1, len(grounded)),
        "answer_prefix_match_rate": sum(answer_match) / max(1, len(answer_match)),
    }


def evaluate(records: list[EvaluationRecord]):
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_list([r.__dict__ for r in records])
        return ragas_evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    except Exception as exc:
        result = local_metrics(records)
        result["fallback_reason"] = str(exc)
        return result
