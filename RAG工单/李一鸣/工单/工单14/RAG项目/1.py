"""Work order 14: recover content from low-quality industrial PDF documents."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, make_chunks, normalize_text


@dataclass
class PageQuality:
    page: int
    text: str
    char_count: int
    replacement_ratio: float
    is_suspect: bool


class OCRBackend(Protocol):
    def recognize(self, image_bytes: bytes) -> str: ...


def assess_page(page_number: int, text: str, minimum_chars: int = 40) -> PageQuality:
    text = normalize_text(text)
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    suspect = len(text) < minimum_chars or replacement_ratio > 0.02
    return PageQuality(page_number, text, len(text), replacement_ratio, suspect)


def repair_ocr_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("|", "I")
    text = re.sub(r"(?<=\d)[Oo](?=\d)", "0", text)
    text = re.sub(r"(?<=\d)[Il](?=\d)", "1", text)
    return normalize_text(text)


def render_suspect_pages(pdf_path: str, pages: list[int], dpi: int = 300) -> dict[int, bytes]:
    import fitz  # type: ignore

    output = {}
    with fitz.open(pdf_path) as document:
        for page_number in pages:
            page = document[page_number - 1]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            output[page_number] = pixmap.tobytes("png")
    return output


def recover_pages(pdf_path: str, text_pages: list[dict[str, Any]], ocr: OCRBackend) -> list[dict[str, Any]]:
    qualities = [assess_page(row["page"], row.get("text", "")) for row in text_pages]
    images = render_suspect_pages(pdf_path, [item.page for item in qualities if item.is_suspect])
    recovered = []
    for quality in qualities:
        text = quality.text
        if quality.is_suspect:
            text = repair_ocr_text(ocr.recognize(images[quality.page]))
        recovered.append({"source": pdf_path, "page": quality.page, "text": text, "metadata": {"ocr_used": quality.is_suspect}})
    return recovered


def industrial_chunks(pages: list[dict[str, Any]]) -> list[Chunk]:
    return make_chunks(pages, chunk_size=650, overlap=100)
