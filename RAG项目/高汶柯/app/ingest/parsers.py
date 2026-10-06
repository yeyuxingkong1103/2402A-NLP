"""多引擎文档解析。

支持：
- PyMuPDF(fitz)：文本层解析（默认）
- pdfplumber：表格解析
- PaddleOCR：扫描件 / 图片文字识别（可选）
- MinerU：复杂版面重排版（可选）
- 多模态大模型：页面图像理解（可选）

按文本层自动选择引擎；重依赖缺失时安全降级。
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile

from app.config import settings
from app.logging_conf import log


# ===== 工具 =====
def _table_to_text(table: list[list]) -> str:
    rows = []
    for row in table or []:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


class BaseParser:
    name = "base"

    def available(self) -> bool:
        return True

    def parse(self, path: str) -> list[dict]:
        raise NotImplementedError


# ===== PyMuPDF =====
class PyMuPDFParser(BaseParser):
    name = "pymupdf"

    def available(self) -> bool:
        try:
            import fitz  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def parse(self, path: str) -> list[dict]:
        import fitz

        pages: list[dict] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                pages.append(
                    {"page": i + 1, "text": page.get_text("text") or "",
                     "images": len(page.get_images(full=True))}
                )
        return pages


# ===== pdfplumber（表格）=====
class PdfPlumberTableParser(BaseParser):
    name = "pdfplumber"

    def available(self) -> bool:
        try:
            import pdfplumber  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def parse(self, path: str) -> list[dict]:
        import pdfplumber

        pages: list[dict] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                table_text = "\n".join(_table_to_text(t) for t in tables)
                if table_text:
                    text = f"{text}\n{table_text}"
                pages.append({"page": i + 1, "text": text, "tables": len(tables)})
        return pages


# ===== PaddleOCR（扫描件 / 图片）=====
class PaddleOCRParser(BaseParser):
    name = "paddleocr"

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def parse(self, path: str) -> list[dict]:
        import fitz
        import numpy as np
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
        pages: list[dict] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=200)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                pages.append({"page": i + 1, "text": self._ocr_image(ocr, img)})
        return pages

    @staticmethod
    def _ocr_image(ocr, img) -> str:
        result = ocr.ocr(img)
        lines: list[str] = []
        for block in result or []:
            for item in block or []:
                # item = [box, (text, score)]
                try:
                    lines.append(item[1][0])
                except Exception:  # noqa: BLE001
                    continue
        return "\n".join(lines)


# ===== MinerU（复杂版面）=====
class MinerUParser(BaseParser):
    name = "mineru"

    def available(self) -> bool:
        return shutil.which("mineru") is not None

    def parse(self, path: str) -> list[dict]:
        with tempfile.TemporaryDirectory() as outdir:
            subprocess.run(
                ["mineru", "-p", path, "-o", outdir], check=True,
                capture_output=True, timeout=1800,
            )
            markdown = ""
            for root, _dirs, files in os.walk(outdir):
                for fn in files:
                    if fn.endswith(".md"):
                        with open(os.path.join(root, fn), encoding="utf-8") as fp:
                            markdown += fp.read() + "\n"
        return [{"page": 1, "text": markdown}]


# ===== 多模态大模型 =====
class MultimodalParser(BaseParser):
    name = "multimodal"

    def available(self) -> bool:
        try:
            import fitz  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def parse(self, path: str) -> list[dict]:
        import fitz

        from app.core.registry import get_llm

        llm = get_llm()
        pages: list[dict] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=144)
                b64 = base64.b64encode(pix.tobytes("png")).decode()
                messages = [
                    {"role": "system", "content": "你是文档解析助手，请完整、准确地转录图片中的文字与表格内容。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "请提取这一页的全部文字内容。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]},
                ]
                try:
                    pages.append({"page": i + 1, "text": llm.chat(messages, temperature=0.0)})
                except Exception as exc:  # noqa: BLE001
                    log.warning("多模态解析第 %d 页失败: %s", i + 1, exc)
                    pages.append({"page": i + 1, "text": page.get_text("text") or ""})
        return pages


PARSERS: dict[str, BaseParser] = {
    "pymupdf": PyMuPDFParser(),
    "pdfplumber": PdfPlumberTableParser(),
    "paddleocr": PaddleOCRParser(),
    "mineru": MinerUParser(),
    "multimodal": MultimodalParser(),
}


def get_parser(name: str) -> BaseParser:
    return PARSERS.get(name, PARSERS["pymupdf"])


# def detect_parser(path: str) -> str:
#     """按文本层密度自动选择解析器：文本充足用 pymupdf，否则 OCR。"""
#     try:
#         import fitz
#
#         with fitz.open(path) as doc:
#             sample = min(3, doc.page_count)
#             total = sum(len(doc[i].get_text("text").strip()) for i in range(sample))
#             avg = total / sample if sample else 0
#         if avg < 50:
#             return "paddleocr" if PARSERS["paddleocr"].available() else "pymupdf"
#         return "pymupdf"
#     except Exception:  # noqa: BLE001
#         return "pymupdf"
def detect_parser(path: str) -> str:
    """统一使用 MinerU 解析 PDF。"""
    return "mineru"


def parse_document(path: str, parser: str = "auto") -> dict:
    """解析文档，返回 {parser, text, pages}。"""
    name = detect_parser(path) if parser in ("auto", "", None) else parser
    p = get_parser(name)
    if not p.available():
        log.warning("解析器 %s 不可用，回退 pymupdf", name)
        p = PARSERS["pymupdf"]
        name = "pymupdf"
    pages = p.parse(path)
    text = "\n".join(pg.get("text", "") for pg in pages)
    log.info("解析 %s: parser=%s pages=%d chars=%d", os.path.basename(path), name, len(pages), len(text))
    return {"parser": name, "text": text, "pages": pages}
