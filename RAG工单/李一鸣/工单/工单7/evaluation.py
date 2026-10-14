"""Small, dependency-free regression evaluator for RAG retrieval and answers."""

from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class EvalCase:
    question: str
    answer: str = ""
    relevant_ids: set[str] = field(default_factory=set)
    required_terms: set[str] = field(default_factory=set)


def hit_at_k(ids: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(ids[:k]) & relevant))


def mrr(ids: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(ids, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(cases: list[EvalCase], retrieve: Callable[[str], list[str]], generate: Callable[[str], str]) -> dict:
    hits = []
    ranks = []
    recalls = []
    for case in cases:
        ids = retrieve(case.question)
        answer = generate(case.question)
        hits.append(hit_at_k(ids, case.relevant_ids, 3))
        ranks.append(mrr(ids, case.relevant_ids))
        terms = case.required_terms
        recalls.append(sum(t.lower() in answer.lower() for t in terms) / max(1, len(terms)))
    return {"hit@3": sum(hits) / max(1, len(hits)), "mrr": sum(ranks) / max(1, len(ranks)), "answer_term_recall": sum(recalls) / max(1, len(recalls))}
