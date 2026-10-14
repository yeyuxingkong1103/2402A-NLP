"""Small dependency-light building blocks shared by the 18 RAG examples.

The individual work-order scripts intentionally keep their domain logic in the
script itself.  This module only contains boring, reusable data and math code
so each example can be copied into a separate project without a large framework.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class Chunk:
    id: str
    text: str
    source: str = ""
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese by character and Latin text by words."""
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def split_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120,
    separators: Sequence[str] = ("\n\n", "\n", "。", "！", "？", ". ", " ", ""),
) -> list[str]:
    """Recursively split text while preferring semantic boundaries."""
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    separator = next((item for item in separators if item and item in text), "")
    parts = list(text) if not separator else text.split(separator)
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + separator + part
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            pieces.append(current.strip())
        current = part
    if current:
        pieces.append(current.strip())
    if len(pieces) == 1 and len(pieces[0]) > chunk_size:
        pieces = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    if overlap <= 0:
        return [p for p in pieces if p]
    merged: list[str] = []
    for index, piece in enumerate(pieces):
        prefix = pieces[index - 1][-overlap:] if index else ""
        merged.append((prefix + "\n" + piece).strip() if prefix else piece)
    return merged


def make_chunks(
    documents: Iterable[dict[str, Any]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    output: list[Chunk] = []
    for doc_index, document in enumerate(documents):
        source = str(document.get("source", document.get("file", "")))
        page = document.get("page")
        metadata = dict(document.get("metadata", {}))
        text = str(document.get("text", ""))
        for chunk_index, part in enumerate(split_text(text, chunk_size, overlap)):
            output.append(
                Chunk(
                    id=f"{doc_index}:{chunk_index}",
                    text=part,
                    source=source,
                    page=page,
                    metadata={**metadata, "chunk_index": chunk_index},
                )
            )
    return output


def hash_embedding(text: str, dimension: int = 384) -> list[float]:
    """Deterministic placeholder embedding for examples and offline demos."""
    vector = [0.0] * dimension
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def dense_search(
    query: str,
    chunks: Sequence[Chunk],
    top_k: int = 8,
    embedder=hash_embedding,
) -> list[tuple[Chunk, float]]:
    query_vector = embedder(query)
    ranked = [(chunk, cosine(query_vector, embedder(chunk.text))) for chunk in chunks]
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]


def lexical_search(query: str, chunks: Sequence[Chunk], top_k: int = 8) -> list[tuple[Chunk, float]]:
    query_tokens = Counter(tokenize(query))
    if not query_tokens:
        return []
    document_tokens = [Counter(tokenize(chunk.text)) for chunk in chunks]
    document_frequency = Counter()
    for tokens in document_tokens:
        document_frequency.update(tokens.keys())
    total = max(1, len(chunks))
    results: list[tuple[Chunk, float]] = []
    for chunk, tokens in zip(chunks, document_tokens):
        score = 0.0
        for token, query_count in query_tokens.items():
            term_frequency = tokens.get(token, 0)
            if not term_frequency:
                continue
            idf = math.log((total + 1) / (document_frequency[token] + 1)) + 1.0
            score += idf * min(query_count, term_frequency)
        if score:
            results.append((chunk, score))
    return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]


def rrf(*ranked_lists: Sequence[tuple[Chunk, float]], k: int = 60, top_k: int = 8) -> list[tuple[Chunk, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    objects: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked, start=1):
            objects[chunk.id] = chunk
            scores[chunk.id] += 1.0 / (k + rank)
    merged = [(objects[item_id], score) for item_id, score in scores.items()]
    return sorted(merged, key=lambda item: item[1], reverse=True)[:top_k]


def reciprocal_rank_fusion(*ranked_lists: Sequence[Chunk], k: int = 60) -> list[Chunk]:
    """RRF variant for callers that only have ordered objects."""
    scores: defaultdict[str, float] = defaultdict(float)
    objects: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            objects[chunk.id] = chunk
            scores[chunk.id] += 1.0 / (k + rank)
    return [objects[item_id] for item_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    import json

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def citation(chunk: Chunk) -> dict[str, Any]:
    return {"source": chunk.source, "page": chunk.page, "chunk_id": chunk.id}


def safe_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
