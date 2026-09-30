import json
import re
from pathlib import Path
from typing import Any

from .text_processing import chunk_text, normalize_text


class DocumentChunker:
    def __init__(self, max_chars: int = 1200, overlap: int = 160, min_chars: int = 80) -> None:
        if min_chars <= 0 or min_chars >= max_chars:
            raise ValueError("min_chars must be positive and smaller than max_chars")
        self.max_chars = max_chars
        self.overlap = overlap
        self.min_chars = min_chars

    def chunk_file(self, source: Path, target: Path) -> dict[str, Any]:
        document = json.loads(source.read_text(encoding="utf-8"))
        chunks = self._chunk_pages(document)
        result = {
            "document_id": document.get("document_id", source.stem),
            "source_path": document.get("source_path", ""),
            "title": document.get("title", source.stem),
            "page_count": document.get("page_count", len(document.get("pages", []))),
            "engines": document.get("engines", ["mineru"]),
            "chunking": {
                "version": "1.0",
                "source": str(source),
                "max_chars": self.max_chars,
                "overlap": self.overlap,
                "min_chars": self.min_chars,
                "strategy": "paragraph-aware page-preserving chunks",
            },
            "chunks": chunks,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def chunk_directory(self, source_dir: Path, target_dir: Path) -> list[dict[str, Any]]:
        target_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for source in sorted(source_dir.glob("*.json")):
            target = target_dir / source.name
            results.append(self.chunk_file(source, target))
        return results

    def _chunk_pages(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        seen: set[str] = set()
        document_id = str(document.get("document_id", ""))
        source = str(document.get("source_path", ""))
        title = str(document.get("title", ""))
        for page in document.get("pages", []):
            page_number = int(page.get("page_number", 0))
            text = normalize_text(str(page.get("mineru_text", "")))
            if not text:
                continue
            page_chunks = self._merge_short_chunks(chunk_text(text, self.max_chars, self.overlap))
            for index, chunk in enumerate(page_chunks, start=1):
                fingerprint = re.sub(r"\W+", "", chunk).casefold()
                if not fingerprint or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                chunks.append(
                    {
                        "chunk_id": f"{document_id}-{page_number}-{index}",
                        "page_start": page_number,
                        "page_end": page_number,
                        "text": chunk,
                        "source": source,
                        "metadata": {
                            "title": title,
                            "page": page_number,
                            "cleaned": True,
                            "chunked": True,
                        },
                    }
                )
        return chunks

    def _merge_short_chunks(self, chunks: list[str]) -> list[str]:
        if len(chunks) < 2:
            return chunks
        merged: list[str] = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue
            combined = f"{merged[-1]}\n\n{chunk}"
            if len(merged[-1]) < self.min_chars and len(combined) <= self.max_chars:
                merged[-1] = combined
            else:
                merged.append(chunk)
        if len(merged) > 1 and len(merged[-1]) < self.min_chars:
            combined = f"{merged[-2]}\n\n{merged[-1]}"
            if len(combined) <= self.max_chars:
                merged[-2] = combined
                merged.pop()
        return merged
