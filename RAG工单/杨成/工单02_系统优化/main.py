"""工单 02：系统优化检索。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.embedding import LocalEmbedding
from common.models import DocumentChunk


@dataclass
class OptimizedResult:
    chunk: DocumentChunk
    score: float
    keyword_score: float
    vector_score: float = 0.0
    citation: dict[str, object] = field(default_factory=dict)


def optimize_results(query: str, chunks: list[DocumentChunk], top_k: int = 5) -> list[OptimizedResult]:
    if top_k <= 0:
        return []
    unique: dict[str, DocumentChunk] = {}
    for chunk in chunks:
        unique.setdefault(chunk.chunk_id, chunk)
    embedding = LocalEmbedding()
    query_vector = embedding.embed(query)
    terms = set(embedding.tokenize(query))
    ranked: list[OptimizedResult] = []
    for chunk in unique.values():
        content_tokens = set(embedding.tokenize(chunk.content))
        keyword_score = sum(term in content_tokens for term in terms) / max(len(terms), 1)
        vector_score = embedding.cosine_similarity(query_vector, embedding.embed(chunk.content))
        score = 0.7 * vector_score + 0.3 * keyword_score
        ranked.append(OptimizedResult(
            chunk=chunk,
            score=score,
            keyword_score=keyword_score,
            vector_score=vector_score,
            citation={"source": chunk.source, "page": chunk.page},
        ))
    ranked.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
    return ranked[:top_k]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="系统优化检索模块")
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
