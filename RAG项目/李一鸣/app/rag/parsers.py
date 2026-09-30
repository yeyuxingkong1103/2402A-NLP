import csv
import io
import logging
from pathlib import Path

from app.rag.types import ParsedDocument

logger = logging.getLogger(__name__)


class DocumentParser:
    def parse(self, file_path: str | Path) -> ParsedDocument:
        # 根据扩展名选择解析器；统一返回 ParsedDocument，后续分块无需关心文件格式。
        path = Path(file_path)
        suffix = path.suffix.lower()
        logger.info("document parsing started: %s", path.name)
        if suffix == ".pdf":
            return self._parse_pdf(path)
        if suffix in {".txt", ".md", ".markdown", ".csv", ".json"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(text=text, pages=[text], metadata={"parser": "text"})
        raise ValueError(f"unsupported document type: {suffix}")

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        # PDF 先尝试文本和表格解析，文本过少时再尝试 OCR。
        pages: list[str] = []
        parser_names: list[str] = []

        try:
            import fitz

            with fitz.open(path) as pdf:
                pages = [page.get_text("text") or "" for page in pdf]
                parser_names.append("pymupdf")
            logger.info("pdf text extracted with PyMuPDF: %s pages", len(pages))
        except ImportError:
            logger.warning("PyMuPDF is not installed; trying pdfplumber")
        except Exception:
            logger.exception("PyMuPDF parsing failed: %s", path)

        tables: list[str] = []
        # pdfplumber 负责补充表格内容；即使 PyMuPDF 已经成功，也仍会尝试提取表格。
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                plumber_pages = []
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    plumber_pages.append(text)
                    for table_number, table in enumerate(page.extract_tables() or [], start=1):
                        table_text = self._table_to_text(table)
                        if table_text:
                            tables.append(
                                f"[Table page={page_number} index={table_number}]\n{table_text}"
                            )
                if not pages or sum(map(len, pages)) < 80:
                    pages = plumber_pages
                    parser_names.append("pdfplumber")
            logger.info("pdf tables extracted: %s tables", len(tables))
        except ImportError:
            logger.warning("pdfplumber is not installed; table extraction skipped")
        except Exception:
            logger.exception("pdfplumber table extraction failed: %s", path)

        text = "\n\n".join(f"[Page {i}]\n{page}" for i, page in enumerate(pages, start=1))
        if tables:
            text = f"{text}\n\n" + "\n\n".join(tables)

        if len(text.strip()) < 80:
            # 扫描 PDF 通常没有文本层，交给可选的 PaddleOCR-VL。
            ocr_text = self._try_ocr(path)
            if ocr_text:
                text = ocr_text
                parser_names.append("paddleocr-vl")

        if not text.strip():
            raise ValueError("PDF contains no extractable text; enable PaddleOCR-VL or provide OCR output")
        return ParsedDocument(
            text=text,
            pages=pages,
            metadata={
                "parser": "+".join(parser_names) or "unknown",
                "page_count": len(pages),
                "table_count": len(tables),
                "ocr_used": "paddleocr-vl" in parser_names,
            },
        )

    @staticmethod
    def _table_to_text(table: list[list[str | None]]) -> str:
        # 用 CSV 格式表达表格，既保留列结构，也方便后续文本分块。
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        for row in table:
            writer.writerow([(cell or "").strip() for cell in row])
        return output.getvalue().strip()

    def _try_ocr(self, path: Path) -> str:
        """Best-effort OCR hook; heavy Paddle dependencies remain optional."""
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except ImportError:
            logger.warning("PaddleOCR-VL is not installed; OCR fallback unavailable")
            return ""
        try:
            ocr = PaddleOCRVL()
            result = ocr.predict(input=str(path))
            parts: list[str] = []
            for item in result:
                if isinstance(item, dict):
                    parts.append(str(item.get("rec_texts", item.get("text", ""))))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        except Exception:
            logger.exception("PaddleOCR-VL parsing failed: %s", path)
            return ""
