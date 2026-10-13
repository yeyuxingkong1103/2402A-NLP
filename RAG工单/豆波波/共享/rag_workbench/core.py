"""Reusable offline RAG primitives; production adapters can replace each function."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Any


@dataclass
class Document:
    id: str
    text: str
    source: str = ""
    page: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self):
        return asdict(self)


def read_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            return "\n".join((page.extract_text() or "") for page in PdfReader(str(path), strict=False).pages)
        except Exception:
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, source: str = "", size: int = 800, overlap: int = 120) -> list[Document]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        value = text[start:end].strip()
        if value:
            chunks.append(Document(id=hashlib.sha1(f"{source}:{start}".encode()).hexdigest()[:16], text=value, source=source))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_./+-]+|[\u4e00-\u9fff]", text.lower())


def _score(query: str, text: str) -> float:
    q = Counter(_tokens(query))
    d = Counter(_tokens(text))
    if not q or not d:
        return 0.0
    overlap = sum(min(q[token], d[token]) for token in q)
    return overlap / math.sqrt(sum(v * v for v in q.values()) * sum(v * v for v in d.values()))


def lexical_search(query: str, documents: Iterable[Document], top_k: int = 5) -> list[dict[str, Any]]:
    rows = [{"document": doc.to_dict(), "score": _score(query, doc.text), "method": "lexical"} for doc in documents]
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]


def hybrid_search(query: str, documents: Iterable[Document], top_k: int = 5, vector_weight: float = 0.7) -> list[dict[str, Any]]:
    if not 0 <= vector_weight <= 1:
        raise ValueError("vector_weight must be between 0 and 1")
    rows = []
    q_tokens = set(_tokens(query))
    for doc in documents:
        lexical = _score(query, doc.text)
        dense_proxy = len(q_tokens & set(_tokens(doc.text))) / max(len(q_tokens), 1)
        rows.append({"document": doc.to_dict(), "score": vector_weight * dense_proxy + (1 - vector_weight) * lexical, "lexical_score": lexical, "dense_score": dense_proxy, "method": "hybrid"})
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]


def build_graph(documents: Iterable[Document]) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    for doc in documents:
        terms = [token for token in _tokens(doc.text) if len(token) > 1]
        unique = list(dict.fromkeys(terms))
        for term in unique:
            entities.setdefault(term, {"id": term, "type": "term", "sources": []})["sources"].append(doc.id)
        for left, right in zip(unique, unique[1:]):
            edges.append({"source": left, "target": right, "document": doc.id})
    return {"entities": list(entities.values()), "edges": edges}


def graph_search(query: str, graph: dict[str, Any], documents: Iterable[Document], top_k: int = 5) -> list[dict[str, Any]]:
    query_terms = set(_tokens(query))
    related = {edge["target"] for edge in graph.get("edges", []) if edge["source"] in query_terms} | query_terms
    rows = []
    for doc in documents:
        terms = set(_tokens(doc.text))
        score = len(terms & related) / max(len(related), 1)
        rows.append({"document": doc.to_dict(), "score": score, "method": "graph"})
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]
