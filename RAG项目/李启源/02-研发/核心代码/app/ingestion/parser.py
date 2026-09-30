"""Public document parser facade.

Models and cleanup helpers remain re-exported here so existing imports keep
working after format-specific adapters were moved into focused modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.parser_adapters import (
    MinerUAdapter,
    PaddleOcrAdapter,
    parse_docx_file,
    parse_jsonl_file,
    parse_pdf_pdfplumber,
    parse_pdf_pymupdf,
    parse_text_file,
)
from app.ingestion.parser_models import (
    DocumentParseError,
    OcrAdapter,
    ParsedDocument,
    ParsedPage,
    QualityReport,
    build_quality_report,
    clean_text,
    file_sha256,
    strip_cross_page_boilerplate,
    validate_quality,
)

logger = logging.getLogger(__name__)
_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown", ".jsonl"}
_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".jsonl": "application/jsonl",
}

# Private aliases are retained for callers that used the old module internals.
_sha256 = file_sha256
_parse_text_file = parse_text_file
_parse_jsonl_file = parse_jsonl_file
_parse_docx_file = parse_docx_file
_parse_pdf_pymupdf = parse_pdf_pymupdf
_parse_pdf_pdfplumber = parse_pdf_pdfplumber


def _select_parser(source: Path, parser: str) -> str:
    selected = parser.lower()
    if selected != "auto":
        return selected
    if source.suffix.lower() in {".txt", ".md", ".markdown"}:
        return "text"
    if source.suffix.lower() == ".jsonl":
        return "jsonl"
    if source.suffix.lower() == ".docx":
        return "docx"
    return "pymupdf"


def _parse_pages(
    source: Path,
    selected_parser: str,
    ocr_adapter: OcrAdapter | None,
) -> list[ParsedPage]:
    """Dispatch to an adapter without importing optional parser dependencies early."""
    if selected_parser == "text":
        return parse_text_file(source)
    if selected_parser == "jsonl":
        return parse_jsonl_file(source)
    if selected_parser == "docx":
        return parse_docx_file(source)
    if selected_parser == "pymupdf":
        return parse_pdf_pymupdf(source)
    if selected_parser == "pdfplumber":
        return parse_pdf_pdfplumber(source)
    if selected_parser == "paddleocr":
        return (ocr_adapter or PaddleOcrAdapter()).parse_pdf(source)
    if selected_parser == "mineru":
        return (ocr_adapter or MinerUAdapter()).parse_pdf(source)
    raise DocumentParseError(f"unsupported parser: {selected_parser}")


def parse_document(
    path: str | Path,
    *,
    parser: str = "auto",
    ocr_adapter: OcrAdapter | None = None,
    quality_min_characters: int = 20,
) -> ParsedDocument:
    """Parse a supported document and return normalized, validated pages.

    Optional OCR dependencies are loaded only when their parser is selected. JSONL
    pages skip cross-page boilerplate removal because repeated customer answers are
    legitimate independent records rather than PDF headers or footers.
    """
    source = Path(path)
    if not source.is_file():
        raise DocumentParseError(f"document does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise DocumentParseError(f"unsupported document type: {source.suffix}")

    selected_parser = _select_parser(source, parser)
    pages = _parse_pages(source, selected_parser, ocr_adapter)
    if selected_parser != "jsonl":
        pages = strip_cross_page_boilerplate(pages)

    document = ParsedDocument(
        source_path=str(source.resolve()),
        file_name=source.name,
        file_sha256=file_sha256(source),
        mime_type=_MIME_TYPES.get(suffix, "text/plain"),
        pages=tuple(page for page in pages if page.text.strip()),
        metadata={"parser": selected_parser},
    )
    quality = validate_quality(document, min_characters=quality_min_characters)
    logger.info(
        "document parsed",
        extra={
            "file_name": document.file_name,
            "file_sha256": document.file_sha256,
            "page_count": quality.page_count,
            "character_count": quality.character_count,
            "duplicate_line_ratio": quality.duplicate_line_ratio,
        },
    )
    return document


__all__ = [
    "DocumentParseError",
    "MinerUAdapter",
    "OcrAdapter",
    "PaddleOcrAdapter",
    "ParsedDocument",
    "ParsedPage",
    "QualityReport",
    "build_quality_report",
    "clean_text",
    "parse_document",
    "strip_cross_page_boilerplate",
    "validate_quality",
]
