import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from .text_processing import chunk_text, document_id, normalize_text

IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
IMAGE_HTML_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d{1,4}$", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t\xa0]+")


class DocumentCleaner:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 160) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_file(self, source: Path, target: Path) -> dict[str, Any]:
        payload = json.loads(source.read_text(encoding="utf-8"))
        pages = payload.get("pages", [])
        raw_texts = [str(page.get("mineru_text", "")) for page in pages]
        basic_texts, basic_stats = self._clean_pages(raw_texts)
        noise_lines = self._find_edge_noise(basic_texts)
        cleaned_texts, edge_stats = self._remove_edge_noise(basic_texts, noise_lines)

        cleaned_pages = []
        for page, text in zip(pages, cleaned_texts):
            cleaned_page = dict(page)
            cleaned_page["mineru_text"] = text
            cleaned_pages.append(cleaned_page)

        chunks, duplicate_count = self._build_chunks(payload, cleaned_pages)
        result = dict(payload)
        result["pages"] = cleaned_pages
        result["chunks"] = chunks
        result["cleaning"] = {
            "version": "1.0",
            "source": "mineru",
            "removed_image_placeholders": basic_stats["image_placeholders"],
            "removed_standalone_page_numbers": basic_stats["page_numbers"],
            "removed_repeated_edge_lines": edge_stats,
            "removed_duplicate_chunks": duplicate_count,
            "nonempty_pages": sum(bool(text) for text in cleaned_texts),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def clean_directory(self, source_dir: Path, target_dir: Path) -> list[dict[str, Any]]:
        target_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for source in sorted(source_dir.glob("*.json")):
            target = target_dir / source.name
            results.append(self.clean_file(source, target))
        return results

    def _clean_pages(self, texts: list[str]) -> tuple[list[str], dict[str, int]]:
        stats = {"image_placeholders": 0, "page_numbers": 0}
        cleaned = []
        for text in texts:
            image_matches = IMAGE_MARKDOWN_RE.findall(text) + IMAGE_HTML_RE.findall(text)
            stats["image_placeholders"] += len(image_matches)
            text = IMAGE_MARKDOWN_RE.sub("", text)
            text = IMAGE_HTML_RE.sub("", text)
            text = unicodedata.normalize("NFKC", text)
            text = text.replace("\x00", " ").replace("﻿", "")
            lines = []
            previous_key = ""
            for raw_line in text.splitlines():
                line = SPACE_RE.sub(" ", raw_line).strip()
                if not line:
                    if lines and lines[-1] != "":
                        lines.append("")
                    continue
                if PAGE_NUMBER_RE.fullmatch(line):
                    stats["page_numbers"] += 1
                    continue
                line_key = self._line_key(line)
                if line_key and line_key == previous_key:
                    continue
                lines.append(line)
                previous_key = line_key
            cleaned.append(normalize_text("\n".join(lines)))
        return cleaned, stats

    def _find_edge_noise(self, texts: list[str]) -> set[str]:
        if len(texts) < 3:
            return set()
        occurrences: dict[str, set[int]] = {}
        representatives: dict[str, str] = {}
        for page_index, text in enumerate(texts):
            lines = [line for line in text.splitlines() if line.strip()]
            edge_lines = set(lines[:3] + lines[-3:])
            for line in edge_lines:
                key = self._line_key(line)
                if not key or len(line) > 120 or self._looks_like_url(line):
                    continue
                occurrences.setdefault(key, set()).add(page_index)
                representatives[key] = line
        threshold = max(3, math.ceil(len(texts) * 0.25))
        return {
            key
            for key, pages in occurrences.items()
            if len(pages) >= threshold and representatives[key]
        }

    def _remove_edge_noise(
        self, texts: list[str], noise_lines: set[str]
    ) -> tuple[list[str], int]:
        if not noise_lines:
            return texts, 0
        kept_noise: set[str] = set()
        removed = 0
        cleaned = []
        for text in texts:
            lines = text.splitlines()
            meaningful_indices = [index for index, line in enumerate(lines) if line.strip()]
            edge_indices = set(meaningful_indices[:3] + meaningful_indices[-3:])
            output = []
            for index, line in enumerate(lines):
                key = self._line_key(line)
                if key in noise_lines and index in edge_indices:
                    if key in kept_noise:
                        removed += 1
                        continue
                    kept_noise.add(key)
                output.append(line)
            cleaned.append(normalize_text("\n".join(output)))
        return cleaned, removed

    def _build_chunks(
        self, payload: dict[str, Any], pages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        chunks: list[dict[str, Any]] = []
        seen: set[str] = set()
        duplicate_count = 0
        source = str(payload.get("source_path", ""))
        doc_id = str(payload.get("document_id")) or document_id(Path(source))
        title = str(payload.get("title", ""))
        for page in pages:
            page_number = int(page.get("page_number", 0))
            for index, text in enumerate(
                chunk_text(str(page.get("mineru_text", "")), self.chunk_size, self.chunk_overlap),
                start=1,
            ):
                fingerprint = re.sub(r"\W+", "", text).casefold()
                if fingerprint in seen:
                    duplicate_count += 1
                    continue
                seen.add(fingerprint)
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}-{page_number}-{index}",
                        "page_start": page_number,
                        "page_end": page_number,
                        "text": text,
                        "source": source,
                        "metadata": {"title": title, "page": page_number, "cleaned": True},
                    }
                )
        return chunks, duplicate_count

    @staticmethod
    def _line_key(line: str) -> str:
        return re.sub(r"\s+", " ", line).strip().casefold()

    @staticmethod
    def _looks_like_url(line: str) -> bool:
        return line.startswith(("http://", "https://", "www."))
