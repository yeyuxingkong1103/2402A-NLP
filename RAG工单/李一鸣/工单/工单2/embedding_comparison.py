"""工单2：比较 bge-large、nomic-embed-text、m3e-large 的检索效果。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from config import MODELS, RetrievalConfig


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model, self.base_url = model, base_url.rstrip("/")

    def _embed(self, text: str) -> list[float]:
        import requests

        response = requests.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text}, timeout=60)
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class BenchmarkResult:
    model: str
    elapsed_seconds: float
    hit_at_3: float
    mean_reciprocal_rank: float


def cosine(left: list[float], right: list[float]) -> float:
    import math

    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right)) or 1.0
    return sum(x * y for x, y in zip(left, right)) / denominator


def benchmark(model_name: str, documents: list[dict[str, Any]], questions: list[dict[str, Any]], embedder: Embedder) -> BenchmarkResult:
    started = time.perf_counter()
    vectors = embedder.embed_documents([doc["text"] for doc in documents])
    hits, reciprocal = 0, 0.0
    for question in questions:
        query_vector = embedder.embed_query(question["question"])
        ranked = sorted(enumerate(vectors), key=lambda item: cosine(query_vector, item[1]), reverse=True)
        relevant = set(question.get("relevant_indices", []))
        for rank, (index, _) in enumerate(ranked, 1):
            if index in relevant:
                if rank <= 3:
                    hits += 1
                reciprocal += 1 / rank
                break
    total = max(1, len(questions))
    return BenchmarkResult(model_name, time.perf_counter() - started, hits / total, reciprocal / total)


def run_comparison(documents_path: str, questions_path: str, output_path: str) -> None:
    documents = json.loads(Path(documents_path).read_text(encoding="utf-8"))
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    results = []
    for name, settings in MODELS.items():
        embedder = OllamaEmbedder(settings["model"])
        results.append(asdict(benchmark(name, documents, questions, embedder)))
    Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("documents")
    parser.add_argument("questions")
    parser.add_argument("--output", default="embedding_comparison.json")
    args = parser.parse_args()
    run_comparison(args.documents, args.questions, args.output)
