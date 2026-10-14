"""人工智能NLP-RAG-混合检索任务：dense + inverted index + RRF + rerank。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Record:
    id: str
    title: str
    text: str
    vector: list[float]
    metadata: dict


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


class InvertedIndex:
    def __init__(self, records: list[Record]):
        self.records = records
        self.postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        self.document_frequency = Counter()
        for index, record in enumerate(records):
            terms = Counter(tokenize(record.title + " " + record.text))
            for term, frequency in terms.items():
                self.postings[term].append((index, frequency))
                self.document_frequency[term] += 1

    def search(self, query: str, top_k: int = 20) -> list[tuple[Record, float]]:
        scores = Counter()
        total = max(1, len(self.records))
        for term in tokenize(query):
            idf = math.log(1 + (total - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5))
            for index, frequency in self.postings.get(term, []):
                scores[index] += idf * (frequency * 2.2) / (frequency + 1.2)
        return [(self.records[index], score) for index, score in scores.most_common(top_k)]


def cosine(a: list[float], b: list[float]) -> float:
    denominator = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)) or 1.0
    return sum(x*y for x, y in zip(a, b)) / denominator


class HybridRetriever:
    def __init__(self, records: list[Record], embedder: Embedder, reranker: Reranker | None = None):
        self.records, self.embedder, self.reranker = records, embedder, reranker
        self.index = InvertedIndex(records)

    def search(self, query: str, top_k: int = 8, dense_weight: float = 0.55):
        query_vector = self.embedder.embed_query(query)
        dense = sorted(((record, cosine(query_vector, record.vector)) for record in self.records), key=lambda item: item[1], reverse=True)[:20]
        sparse = self.index.search(query, 20)
        fused: defaultdict[str, float] = defaultdict(float)
        records = {}
        for weight, ranking in ((dense_weight, dense), (1 - dense_weight, sparse)):
            for rank, (record, _score) in enumerate(ranking, 1):
                fused[record.id] += weight / (60 + rank)
                records[record.id] = record
        candidates = sorted(((records[key], score) for key, score in fused.items()), key=lambda item: item[1], reverse=True)[:20]
        if self.reranker and candidates:
            rerank_scores = self.reranker.rerank(query, [item[0].title + "\n" + item[0].text for item in candidates])
            candidates = [(record, score) for (record, _), score in zip(candidates, rerank_scores)]
            candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]
