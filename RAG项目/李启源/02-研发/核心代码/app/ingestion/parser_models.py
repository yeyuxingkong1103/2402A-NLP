"""Shared document models, cleanup rules, and quality checks.

Keeping normalization and quality policy separate from format adapters makes the
parser facade small while preserving one validation contract for every format.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

_WATERMARK_PATTERNS = (
    re.compile(r"^\s*第\s*\d+\s*页\s*/\s*共\s*\d+\s*页\s*$"),
    re.compile(r"^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(扫描件|内部资料|仅供参考|confidential)\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-_=]{4,}\s*$"),
)
_SENTENCE_END = "。！？.!?；;："


class DocumentParseError(RuntimeError):
    """Raised when a document cannot be parsed or validated."""


class OcrAdapter(Protocol):
    """Boundary implemented by optional scanned-PDF parsers."""

    def parse_pdf(self, path: Path) -> list["ParsedPage"]:
        """Return OCR pages extracted from a PDF file."""


@dataclass(slots=True, frozen=True)
class ParsedPage:
    """Normalized text and metadata for one logical document page."""

    page_number: int
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ParsedDocument:
    """Normalized document returned by every parser implementation."""

    source_path: str
    file_name: str
    file_sha256: str
    mime_type: str
    pages: tuple[ParsedPage, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Join non-empty pages while retaining paragraph boundaries."""
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def character_count(self) -> int:
        """Return the number of normalized characters."""
        return len(self.text)


@dataclass(slots=True, frozen=True)
class QualityReport:
    """Lightweight document signals checked before indexing."""

    character_count: int
    page_count: int
    duplicate_line_ratio: float
    average_page_characters: float


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally so large uploads are not read twice into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _normalize_line(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line.replace("\x00", " ")).strip()


def _looks_like_prose(line: str) -> bool:
    """Distinguish repeated prose from short header/footer boilerplate."""
    return line.endswith(tuple(_SENTENCE_END))


def clean_text(text: str) -> str:
    """Normalize whitespace and conservatively remove watermark lines.

    Empty lines are deliberately retained because paragraph chunking uses them as
    semantic boundaries. Repeated prose is also retained; only short lines that
    look like boilerplate are removed at this per-page stage.
    """
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if line and any(pattern.match(line) for pattern in _WATERMARK_PATTERNS):
            continue
        kept.append(line)
    if not any(kept):
        return ""

    line_counts: dict[str, int] = {}
    for line in kept:
        if line:
            line_counts[line] = line_counts.get(line, 0) + 1
    repeated = {
        line
        for line, count in line_counts.items()
        if count >= 3 and len(line) <= 120 and not _looks_like_prose(line)
    }
    cleaned = "\n".join(line for line in kept if line not in repeated)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def strip_cross_page_boilerplate(
    pages: Sequence[ParsedPage], *, min_pages: int = 3, max_line_length: int = 120
) -> list[ParsedPage]:
    """Drop short lines repeated on several distinct pages.

    Cross-page repetition belongs here rather than in ``clean_text`` because a
    single-page cleaner cannot safely tell a footer from legitimate repeated text.
    """
    if len(pages) < min_pages:
        return list(pages)
    occurrences: dict[str, set[int]] = {}
    for page in pages:
        for line in {item for item in page.text.splitlines() if item}:
            if len(line) <= max_line_length:
                occurrences.setdefault(line, set()).add(page.page_number)
    boilerplate = {
        line for line, page_numbers in occurrences.items()
        if len(page_numbers) >= min_pages
    }
    if not boilerplate:
        return list(pages)
    return [
        ParsedPage(
            page_number=page.page_number,
            text=re.sub(
                r"\n{3,}",
                "\n\n",
                "\n".join(
                    line for line in page.text.splitlines() if line not in boilerplate
                ),
            ).strip(),
            title=page.title,
            metadata=page.metadata,
        )
        for page in pages
    ]


def build_quality_report(document: ParsedDocument) -> QualityReport:
    """Compute filtering and logging signals without changing document content."""
    lines = [line.strip() for line in document.text.splitlines() if line.strip()]
    duplicate_count = len(lines) - len(set(lines))
    page_count = len(document.pages)
    return QualityReport(
        character_count=document.character_count,
        page_count=page_count,
        duplicate_line_ratio=duplicate_count / len(lines) if lines else 0.0,
        average_page_characters=(
            document.character_count / page_count if page_count else 0.0
        ),
    )


def validate_quality(
    document: ParsedDocument,
    *,
    min_characters: int = 20,
    max_duplicate_line_ratio: float = 0.65,
) -> QualityReport:
    """Reject empty, OCR-failed, or obviously unusable extracted documents."""
    report = build_quality_report(document)
    if report.page_count == 0 or report.character_count < min_characters:
        raise DocumentParseError(
            f"document contains insufficient text: {report.character_count} characters"
        )
    if report.duplicate_line_ratio > max_duplicate_line_ratio:
        raise DocumentParseError(
            "document duplicate line ratio is too high: "
            f"{report.duplicate_line_ratio:.2f}"
        )
    return report
