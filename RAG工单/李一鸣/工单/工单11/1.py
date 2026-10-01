"""Work order 11: prepare and fine-tune a domain embedding model."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class TrainingPair:
    query: str
    positive: str
    hard_negative: str = ""


def build_pairs(rows: Iterable[dict], negative_pool: list[str] | None = None) -> list[TrainingPair]:
    negative_pool = negative_pool or []
    pairs = []
    for row in rows:
        query = row.get("query", row.get("question", "")).strip()
        positive = row.get("positive", row.get("passage", row.get("answer", ""))).strip()
        negative = row.get("hard_negative", "")
        if not negative and negative_pool:
            candidates = [item for item in negative_pool if item != positive]
            negative = random.choice(candidates) if candidates else ""
        if query and positive:
            pairs.append(TrainingPair(query, positive, negative))
    return pairs


def save_pairs(pairs: Iterable[TrainingPair], path: str) -> None:
    Path(path).write_text("\n".join(json.dumps(pair.__dict__, ensure_ascii=False) for pair in pairs), encoding="utf-8")


def train_embedding_model(train_path: str, output_dir: str, base_model: str = "BAAI/bge-m3") -> None:
    """Sentence-Transformers training entry point; hard negatives are optional."""
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
    from datasets import Dataset

    rows = [json.loads(line) for line in Path(train_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    data = Dataset.from_list(rows)
    model = SentenceTransformer(base_model)
    loss = losses.MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(model=model, train_dataset=data, loss=loss)
    trainer.train()
    model.save(output_dir)


def evaluate_model(model, cases: list[TrainingPair], k: int = 5) -> dict[str, float]:
    queries = model.encode([case.query for case in cases], normalize_embeddings=True)
    passages = model.encode([case.positive for case in cases], normalize_embeddings=True)
    scores = queries @ passages.T
    hits = 0
    for row, expected in zip(scores, range(len(cases))):
        hits += int(expected in row.argsort()[::-1][:k])
    return {f"recall@{k}": hits / max(1, len(cases))}
