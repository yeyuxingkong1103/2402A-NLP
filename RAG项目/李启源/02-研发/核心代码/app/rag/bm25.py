"""Small dependency-free BM25 scorer used by sparse retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

_CJK_RUN = re.compile(r"[一-鿿㐀-䶿]+")
_ASCII_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese runs into unigrams/bigrams and keep ASCII words."""
    tokens: list[str] = []
    for part in re.findall(r"[一-鿿㐀-䶿]+|[a-z0-9]+", text.lower()):
        if _ASCII_WORD.fullmatch(part):
            tokens.append(part)
        else:
            tokens.extend(part)
            tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
    return tokens


class BM25Scorer:
    """Score a query against one or more candidate documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        query_keywords: Iterable[str],
        document_text: str,
        *,
        documents: Iterable[str] | None = None,
    ) -> float:
        """Return BM25 using corpus IDF when a candidate corpus is supplied."""
        query = tokenize(" ".join(query_keywords))
        document = tokenize(document_text)
        if not query or not document:
            return 0.0
        corpus = [tokenize(item) for item in documents] if documents is not None else [document]
        avg_length = sum(len(item) for item in corpus) / max(len(corpus), 1)
        document_frequency = {
            token: sum(token in item for item in corpus) for token in set(query)
        }
        total_documents = len(corpus)
        counts = Counter(document)
        score = 0.0
        for token in query:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            idf = math.log(1 + (total_documents - document_frequency[token] + 0.5) /
                           (document_frequency[token] + 0.5))
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * len(document) / max(avg_length, 1)
            )
            score += idf * frequency * (self.k1 + 1) / denominator
        return score
