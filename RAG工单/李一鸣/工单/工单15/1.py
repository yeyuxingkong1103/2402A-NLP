"""Work order 15: cross-modal retrieval for technical drawings and text."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, dense_search, lexical_search, rrf


@dataclass
class DrawingRecord:
    id: str
    source: str
    page: int
    image_path: str
    caption: str
    OCR: str = ""
    entities: list[str] | None = None


class MultimodalEncoder(Protocol):
    def encode_text(self, text: str) -> list[float]: ...
    def encode_image(self, image_path: str) -> list[float]: ...


class CrossModalRetriever:
    def __init__(self, text_chunks: list[Chunk], drawings: list[DrawingRecord], encoder: MultimodalEncoder | None = None):
        self.text_chunks = text_chunks
        self.drawings = drawings
        self.encoder = encoder

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        text_hits = rrf(
            dense_search(query, self.text_chunks, top_k * 2),
            lexical_search(query, self.text_chunks, top_k * 2),
            top_k=top_k,
        )
        results = [{"kind": "text", "record": chunk.to_dict(), "score": score} for chunk, score in text_hits]
        terms = set(query.lower().split())
        for drawing in self.drawings:
            searchable = f"{drawing.caption} {drawing.OCR} {' '.join(drawing.entities or [])}".lower()
            lexical_score = sum(term in searchable for term in terms) / max(1, len(terms))
            image_score = 0.0
            if self.encoder:
                from common import cosine

                image_score = cosine(self.encoder.encode_text(query), self.encoder.encode_image(drawing.image_path))
            score = 0.55 * image_score + 0.45 * lexical_score
            if score > 0:
                results.append({"kind": "drawing", "record": drawing.__dict__, "score": score})
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def drawing_index_text(drawing: DrawingRecord) -> str:
    return "\n".join([drawing.caption, drawing.OCR, *(drawing.entities or [])]).strip()
