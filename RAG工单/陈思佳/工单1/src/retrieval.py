from collections import Counter
import math
import re
from dataclasses import dataclass
from typing import Iterable

from rag_core import Chunk


TOKEN_PATTERN = re.compile(r"[一-鿿]|[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    highlights: tuple[str, ...] = ()


class BM25Retriever:
    """Small dependency-free BM25 retriever for a local RAG baseline."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.term_frequency = [Counter(tokens) for tokens in self.tokens]
        self.document_frequency = Counter(
            token for tokens in self.tokens for token in set(tokens)
        )
        self.average_length = (
            sum(len(tokens) for tokens in self.tokens) / len(self.tokens)
            if self.tokens
            else 0
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []
        total_documents = len(self.chunks)
        scored: list[SearchResult] = []
        for index, frequencies in enumerate(self.term_frequency):
            length = len(self.tokens[index])
            score = 0.0
            for token in query_tokens:
                if not frequencies[token]:
                    continue
                idf = math.log(1 + (total_documents - self.document_frequency[token] + 0.5) / (self.document_frequency[token] + 0.5))
                denominator = frequencies[token] + self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1))
                score += idf * frequencies[token] * (self.k1 + 1) / denominator
            if score > 0:
                scored.append(SearchResult(self.chunks[index], score, tuple(query_tokens)))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]
