"""Work order 02: improve PDF RAG with quality-aware parsing and reranking."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, dense_search, lexical_search, make_chunks, rrf


@dataclass
class ParsingConfig:
    chunk_size: int = 760
    overlap: int = 120
    remove_headers: bool = True
    remove_footers: bool = True
    min_text_length: int = 20


def clean_page_text(text: str, config: ParsingConfig) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if config.remove_headers and lines:
        lines = lines[1:]
    if config.remove_footers and lines:
        lines = lines[:-1]
    text = "\n".join(lines)
    text = re.sub(r"第\s*\d+\s*页|Page\s+\d+", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_pages(pages: list[dict[str, Any]], config: ParsingConfig | None = None) -> list[dict[str, Any]]:
    config = config or ParsingConfig()
    output = []
    for page in pages:
        text = clean_page_text(str(page.get("text", "")), config)
        if len(text) >= config.min_text_length:
            output.append({**page, "text": text})
    return output


class ImprovedPdfRetriever:
    def __init__(self, pages: list[dict[str, Any]], config: ParsingConfig | None = None):
        self.config = config or ParsingConfig()
        self.chunks = make_chunks(prepare_pages(pages, self.config), self.config.chunk_size, self.config.overlap)

    def retrieve(self, query: str, top_k: int = 6) -> list[tuple[Chunk, float]]:
        dense = dense_search(query, self.chunks, top_k=top_k * 3)
        lexical = lexical_search(query, self.chunks, top_k=top_k * 3)
        return rrf(dense, lexical, top_k=top_k)

    def answer_context(self, query: str, top_k: int = 6) -> str:
        hits = self.retrieve(query, top_k)
        return "\n\n".join(f"[{i}] {chunk.text}" for i, (chunk, _) in enumerate(hits, 1))
