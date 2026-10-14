"""Work order 07: functional tests and retrieval/generation evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from common import Chunk, tokenize


@dataclass
class EvalCase:
    question: str
    answer: str
    relevant_chunk_ids: set[str] = field(default_factory=set)
    required_terms: set[str] = field(default_factory=set)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def answer_term_recall(answer: str, required_terms: Iterable[str]) -> float:
    terms = list(required_terms)
    return sum(term.lower() in answer.lower() for term in terms) / max(1, len(terms))


def evaluate_retriever(cases: list[EvalCase], retrieve: Callable[[str], list[Chunk]], ks=(1, 3, 5)) -> dict[str, float]:
    values: dict[str, list[float]] = {f"hit@{k}": [] for k in ks}
    values["mrr"] = []
    for case in cases:
        ids = [chunk.id for chunk in retrieve(case.question)]
        for k in ks:
            values[f"hit@{k}"].append(hit_at_k(ids, case.relevant_chunk_ids, k))
        values["mrr"].append(reciprocal_rank(ids, case.relevant_chunk_ids))
    return {key: sum(items) / max(1, len(items)) for key, items in values.items()}


def evaluate_answers(cases: list[EvalCase], generate: Callable[[str], str]) -> dict[str, float]:
    scores = [answer_term_recall(generate(case.question), case.required_terms) for case in cases]
    return {"answer_term_recall": sum(scores) / max(1, len(scores))}


def run_regression_suite(cases: list[EvalCase], retrieve, generate) -> dict[str, Any]:
    retrieval = evaluate_retriever(cases, retrieve)
    generation = evaluate_answers(cases, generate)
    return {"retrieval": retrieval, "generation": generation, "passed": retrieval["hit@3"] >= 0.7}
