from __future__ import annotations

"""BM25 关键词检索实现，用于补充向量检索的精确匹配能力。"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .filtering import matches_metadata


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


@dataclass(frozen=True)
class KeywordDocument:
    chunk_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class KeywordResult:
    chunk_id: str
    chunk_index: int
    score: float
    content: str
    metadata: dict[str, Any]


def tokenize(text: str) -> list[str]:
    tokens = [match.group(0) for match in _TOKEN_PATTERN.finditer(text.lower())]
    chinese_tokens = [token for token in tokens if "一" <= token <= "鿿"]
    tokens.extend("".join(pair) for pair in zip(chinese_tokens, chinese_tokens[1:]))
    return tokens


class BM25Index:
    def __init__(self, documents: list[KeywordDocument], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokenized = [tokenize(document.content) for document in documents]
        self._document_lengths = [len(tokens) for tokens in self._tokenized]
        self._average_length = sum(self._document_lengths) / len(documents) if documents else 0.0
        self._document_frequency: dict[str, int] = {}
        for tokens in self._tokenized:
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1

    @classmethod
    def from_jsonl(cls, path: Path) -> "BM25Index":
        documents: list[KeywordDocument] = []
        if not path.exists():
            return cls(documents)
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                documents.append(
                    KeywordDocument(
                        chunk_id=str(chunk.get("chunk_id", "")),
                        chunk_index=int(chunk.get("chunk_index", 0)),
                        content=str(chunk.get("content", "")),
                        metadata=dict(chunk.get("metadata", {})),
                    )
                )
        return cls(documents)

    def search(self, query: str, top_k: int, filters: dict[str, str] | None = None) -> list[KeywordResult]:
        if not query.strip() or not self.documents or top_k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        document_count = len(self.documents)
        scores: list[float] = []
        for document, tokens, document_length in zip(
            self.documents, self._tokenized, self._document_lengths, strict=True
        ):
            if not matches_metadata(document.metadata, filters):
                scores.append(0.0)
                continue
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency.get(token, 0)
                inverse_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_factor = 1 - self.b + self.b * document_length / (self._average_length or 1.0)
                score += inverse_frequency * frequency * (self.k1 + 1) / (frequency + self.k1 * length_factor)
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [
            KeywordResult(
                chunk_id=self.documents[index].chunk_id,
                chunk_index=self.documents[index].chunk_index,
                score=score,
                content=self.documents[index].content,
                metadata=self.documents[index].metadata,
            )
            for index, score in ranked[:top_k]
            if score > 0
        ]
