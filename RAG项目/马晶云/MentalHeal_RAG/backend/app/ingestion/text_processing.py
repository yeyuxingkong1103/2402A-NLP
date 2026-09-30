import hashlib
import re
from pathlib import Path


def document_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:24]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("﻿", " ")
    text = re.sub(r"[\t  ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def merge_page_text(native_text: str, ocr_text: str, mineru_text: str, vision_ocr_text: str = "") -> str:
    sources = [normalize_text(value) for value in (native_text, ocr_text, mineru_text, vision_ocr_text)]
    unique: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if not source:
            continue
        fingerprint = re.sub(r"\W+", "", source).casefold()
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(source)
    return "\n\n".join(unique)


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 160) -> list[str]:
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("max_chars must be positive and overlap must be smaller")
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                segment = paragraph[start:end].strip()
                if segment:
                    chunks.append(segment)
                if end == len(paragraph):
                    break
                start = end - overlap
            continue

        combined = f"{current}\n\n{paragraph}" if current else paragraph
        if len(combined) <= max_chars:
            current = combined
        else:
            chunks.append(current)
            tail = current[-overlap:].strip() if overlap else ""
            current = f"{tail}\n\n{paragraph}" if tail else paragraph

    if current:
        chunks.append(current)
    return chunks
