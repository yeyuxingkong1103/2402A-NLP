from dataclasses import dataclass
from typing import Iterable

from retrieval import BM25Retriever, SearchResult, tokenize


@dataclass(frozen=True)
class HybridResult:
    chunk: object
    score: float
    lexical_score: float
    dense_score: float


class HybridRetriever:
    """Blend lexical BM25 scores with an optional embedding similarity function."""

    def __init__(self, chunks: Iterable, embedder=None, lexical_weight: float = 0.65):
        self.chunks = list(chunks)
        self.lexical = BM25Retriever(self.chunks)
        self.embedder = embedder
        self.lexical_weight = lexical_weight
        self.vectors = [embedder.embed(chunk.text) for chunk in self.chunks] if embedder else []

    def search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        lexical_results = self.lexical.search(query, top_k=len(self.chunks))
        lexical_by_index = {result.chunk.index: result.score for result in lexical_results}
        max_lexical = max(lexical_by_index.values(), default=1.0)
        dense_scores: dict[int, float] = {}
        if self.embedder:
            query_vector = self.embedder.embed(query)
            for index, vector in enumerate(self.vectors):
                dense_scores[index] = self.embedder.similarity(query_vector, vector)
        max_dense = max(dense_scores.values(), default=1.0) or 1.0
        results = []
        for index, chunk in enumerate(self.chunks):
            lexical_score = lexical_by_index.get(chunk.index, 0.0) / max_lexical
            dense_score = dense_scores.get(index, 0.0) / max_dense
            score = self.lexical_weight * lexical_score + (1 - self.lexical_weight) * dense_score
            if score > 0:
                results.append(HybridResult(chunk, score, lexical_score, dense_score))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def compress_context(results: Iterable[HybridResult], max_chars: int = 4000) -> str:
    parts: list[str] = []
    length = 0
    for result in results:
        text = result.chunk.text
        if length + len(text) > max_chars:
            break
        parts.append(text)
        length += len(text)
    return "\n\n".join(parts)
