"""Concrete document-format adapters used by :mod:`app.ingestion.parser`."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.ingestion.parser_models import (
    DocumentParseError,
    ParsedPage,
    clean_text,
)


def parse_text_file(path: Path) -> list[ParsedPage]:
    """Parse UTF-8 text or Markdown as one logical page."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError(f"text file is not valid UTF-8: {path.name}") from exc
    return [
        ParsedPage(
            page_number=1,
            text=clean_text(text),
            metadata={"parser": "text", "raw_character_count": len(text)},
        )
    ]


def parse_jsonl_file(path: Path) -> list[ParsedPage]:
    """Map each customer-service JSON object to one retrievable page."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise DocumentParseError(f"JSONL file is not valid UTF-8: {path.name}") from exc

    pages: list[ParsedPage] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise DocumentParseError(
                f"invalid JSON on line {line_number} in {path.name}"
            ) from exc
        if not isinstance(record, dict):
            raise DocumentParseError(
                f"JSONL line {line_number} must contain an object"
            )

        user = record.get("user")
        bot = record.get("bot")
        user_text = user.strip() if isinstance(user, str) else ""
        bot_text = bot.strip() if isinstance(bot, str) else ""
        if not user_text or not bot_text:
            raise DocumentParseError(
                f"JSONL line {line_number} requires non-empty user and bot fields"
            )

        scene_value = record.get("scene")
        tone_value = record.get("tone")
        scene = scene_value.strip() if isinstance(scene_value, str) else ""
        tone = tone_value.strip() if isinstance(tone_value, str) else ""
        category = record.get("category")
        group_id = record.get("question_group_id")
        source_line = record.get("source_line")
        fields = [
            f"场景：{scene}" if scene else "",
            f"用户：{user_text}",
            f"客服：{bot_text}",
            f"语气：{tone}" if tone else "",
        ]
        pages.append(
            ParsedPage(
                page_number=len(pages) + 1,
                text=clean_text("\n".join(field for field in fields if field)),
                metadata={
                    "parser": "jsonl",
                    "record_number": line_number,
                    "source_line": (
                        source_line if isinstance(source_line, int) else line_number
                    ),
                    "scene": scene,
                    "tone": tone,
                    "category": category if isinstance(category, str) else "",
                    "question_group_id": group_id if isinstance(group_id, str) else "",
                },
            )
        )
    return pages


def parse_docx_file(path: Path) -> list[ParsedPage]:
    """Parse DOCX paragraphs as one logical page."""
    try:
        from docx import Document  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DocumentParseError("python-docx is required for DOCX parsing") from exc
    document = Document(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    text = clean_text("\n".join(item for item in paragraphs if item))
    return [
        ParsedPage(
            page_number=1,
            text=text,
            metadata={"parser": "docx", "raw_character_count": len(text)},
        )
    ]


def parse_pdf_pymupdf(path: Path) -> list[ParsedPage]:
    """Extract PDF page text with PyMuPDF."""
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DocumentParseError("PyMuPDF is required for the pymupdf parser") from exc
    pages: list[ParsedPage] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text")
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=clean_text(raw_text),
                    metadata={
                        "parser": "pymupdf",
                        "raw_character_count": len(raw_text),
                    },
                )
            )
    return pages


def parse_pdf_pdfplumber(path: Path) -> list[ParsedPage]:
    """Extract PDF page text with pdfplumber."""
    try:
        import pdfplumber  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DocumentParseError("pdfplumber is required for the pdfplumber parser") from exc
    pages: list[ParsedPage] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=clean_text(raw_text),
                    metadata={
                        "parser": "pdfplumber",
                        "raw_character_count": len(raw_text),
                    },
                )
            )
    return pages


class PaddleOcrAdapter:
    """Optional PaddleOCR adapter for scanned PDFs."""

    def __init__(self, *, lang: str = "ch") -> None:
        self.lang = lang

    def parse_pdf(self, path: Path) -> list[ParsedPage]:
        try:
            import fitz  # type: ignore[import-not-found]
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DocumentParseError(
                "PaddleOCR parsing requires paddleocr, paddlepaddle, and PyMuPDF"
            ) from exc
        ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
        pages: list[ParsedPage] = []
        with fitz.open(path) as pdf, tempfile.TemporaryDirectory() as tmp_dir:
            for page_number, page in enumerate(pdf, start=1):
                image_path = Path(tmp_dir) / f"page-{page_number}.png"
                page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(
                    str(image_path)
                )
                lines: list[str] = []
                for page_result in ocr.ocr(str(image_path), cls=True) or []:
                    for item in page_result or []:
                        if len(item) >= 2 and item[1]:
                            lines.append(str(item[1][0]))
                raw_text = "\n".join(lines)
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=clean_text(raw_text),
                        metadata={
                            "parser": "paddleocr",
                            "raw_character_count": len(raw_text),
                        },
                    )
                )
        return pages


class MinerUAdapter:
    """Run a configured MinerU wrapper that writes normalized JSON output."""

    def __init__(self, command: str | None = None) -> None:
        self.command = command or os.getenv("MINERU_COMMAND")

    def parse_pdf(self, path: Path) -> list[ParsedPage]:
        if not self.command:
            raise DocumentParseError("MinerU requires MINERU_COMMAND")
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            # A fixed local name prevents an untrusted upload name from becoming
            # syntax if an operator deliberately wraps MinerU with ``sh -c``.
            input_path = work_dir / "input.pdf"
            output_path = work_dir / "mineru-output.json"
            shutil.copyfile(path, input_path)
            # Parse the configured template into argv. shell=False prevents this
            # process from interpreting command punctuation as shell syntax.
            try:
                arguments = [
                    item.format(input=str(input_path), output=str(output_path))
                    for item in shlex.split(self.command)
                ]
            except (ValueError, KeyError) as exc:
                raise DocumentParseError("Invalid MINERU_COMMAND template") from exc
            if not arguments:
                raise DocumentParseError("MINERU_COMMAND is empty")
            completed = subprocess.run(
                arguments,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise DocumentParseError(f"MinerU failed: {detail}")
            if not output_path.exists():
                raise DocumentParseError("MinerU command did not create expected JSON output")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            pages = payload.get("pages")
            if isinstance(pages, list):
                return [
                    ParsedPage(
                        page_number=int(item.get("page_number", index + 1)),
                        text=clean_text(str(item.get("text", ""))),
                        title=item.get("title"),
                        metadata={"parser": "mineru"},
                    )
                    for index, item in enumerate(pages)
                    if str(item.get("text", "")).strip()
                ]
            text = clean_text(str(payload.get("markdown") or payload.get("text") or ""))
            return [ParsedPage(page_number=1, text=text, metadata={"parser": "mineru"})]
