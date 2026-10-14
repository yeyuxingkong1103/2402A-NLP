"""Compare Ollama embedding models on the same query/document set."""

from __future__ import annotations

import time
from typing import Iterable

import requests

from config import config


def embed(model: str, text: str) -> list[float]:
    response = requests.post(f"{config.ollama_base_url}/api/embeddings", json={"model": model, "prompt": text}, timeout=60)
    response.raise_for_status()
    return response.json()["embedding"]


def cosine(left: list[float], right: list[float]) -> float:
    import math
    norm = (sum(v * v for v in left) ** 0.5) * (sum(v * v for v in right) ** 0.5) or 1.0
    return sum(a * b for a, b in zip(left, right)) / norm


def compare(query: str, documents: Iterable[str]) -> list[dict]:
    rows = []
    documents = list(documents)
    for model in config.models:
        started = time.perf_counter()
        query_vector = embed(model, query)
        scored = [(index, cosine(query_vector, embed(model, document))) for index, document in enumerate(documents)]
        scored.sort(key=lambda item: item[1], reverse=True)
        rows.append({"model": model, "elapsed_seconds": time.perf_counter() - started, "top_k": scored[:config.top_k]})
    return rows
