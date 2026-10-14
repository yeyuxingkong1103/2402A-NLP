"""Ragas-compatible evaluation record definitions."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class EvaluationRecord:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def save_json(records: list[EvaluationRecord], path: str) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2), encoding="utf-8")


EVALUATION_FIELDS = ("question", "answer", "contexts", "ground_truth")
