"""
pdf_parser.py — PDF 解析

三级解析链路，任一环节缺失自动降级，保证在没装重型组件的机器上也能跑：

    minerU     版面分析，处理双栏、公式、图表标题（可选，config.USE_MINERU）
    PyMuPDF    主力文本抽取（必备）
    pdfplumber 表格抽取，转成 Markdown 表格文本（可选）
    PaddleOCR  扫描件 / 图片内文字识别，优先走 VL 版面分析（可选，config.USE_OCR）

统一输出：[{"text": 文本, "page": 页码(从1开始), "type": "text"|"table"|"image"}]
"""

from __future__ import annotations

import re
from pathlib import Path

import config


def _table_to_text(rows: list[list]) -> str:
    """把 pdfplumber 提取的表格转成 Markdown 表格，保留行列对应关系。"""
    cleaned: list[list[str]] = []
    for row in rows or []:
        cells = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
        if any(cells):
            cleaned.append(cells)

    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    lines = ["| " + " | ".join(cleaned[0]) + " |"]
    lines.append("|" + "---|" * width)
    lines.extend("| " + " | ".join(row) + " |" for row in cleaned[1:])
    return "\n".join(lines)


# 水印判定：短文本在超过一半的页面上、于同一位置重复出现才算数。
# 阈值取 0.5 是让"每页都盖"的水印必定命中，又把偶尔重复的正文排除在外。
_WATERMARK_MAX_LEN = 60
_WATERMARK_PAGE_RATIO = 0.5


def _strip_watermark(doc) -> list[str]:
    """检测并删除水印层，返回逐页清理后的正文。

    水印的特征是"位置固定 + 跨页重复"：页码、"内部资料严禁外传"、
    "第 3 页 / 共 12 页"都属此类。只看单页分辨不出来，必须整本统计——
    在多页同一坐标重复出现的短文本判定为水印，再按行从正文里剔除。
    """
    seen: dict[tuple, set[int]] = {}
    for page_no in range(doc.page_count):
        for block in doc[page_no].get_text("blocks"):
            text = (block[4] or "").strip()
            # block[6] 是块类型，1 表示图片块没有文本；长段落不可能是水印
            if block[6] != 0 or not text or len(text) > _WATERMARK_MAX_LEN:
                continue
            # 坐标量化到 5pt 一格：水印位置存在像素级抖动，不能要求完全相等
            key = (round(block[0] / 5), round(block[1] / 5), text)
            seen.setdefault(key, set()).add(page_no)

    threshold = max(2, doc.page_count * _WATERMARK_PAGE_RATIO)
    marks = {key[2] for key, pages in seen.items() if len(pages) >= threshold}

    texts: list[str] = []
    for page_no in range(doc.page_count):
        kept = [
            line
            for line in doc[page_no].get_text("text").splitlines()
            if line.strip() not in marks
        ]
        texts.append(re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip())
    return texts


def _render_image(doc, img):
    """把内嵌图片渲染成 OCR 能吃的 RGB ndarray，过小或解码失败返回 None。"""
    import numpy as np
    import pymupdf as fitz

    try:
        pix = fitz.Pixmap(doc, img[0])
        if pix.n - pix.alpha >= 4 or pix.n < 3:  # CMYK 与灰度都先转成 RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.width < 100 or pix.height < 100:  # 装饰性小图不值得花 OCR 开销
            return None
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
    except Exception:
        return None


class PDFParser:
    """PDF 解析器。重型组件不可用时自动降级，不会抛异常中断导入。"""

    def __init__(self, use_mineru: bool | None = None, use_ocr: bool | None = None):
        self.use_mineru = config.USE_MINERU if use_mineru is None else use_mineru
        self.use_ocr = config.USE_OCR if use_ocr is None else use_ocr
        self._mineru_failed = False
        self._ocr_engine = None
        self._ocr_failed = False

    # ------------------------------------------------------------ minerU

    def _parse_with_mineru(self, pdf_path: Path) -> list[dict] | None:
        """用 minerU 做版面分析。任何异常都返回 None 让上层降级。"""
        if self._mineru_failed:
            return None

        try:
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod
        except ImportError:
            self._mineru_failed = True
            return None

        try:
            dataset = PymuDocDataset(pdf_path.read_bytes())
            method = dataset.classify()
            infer = (
                doc_analyze(dataset, ocr=True)
                if method == SupportedPdfParseMethod.OCR
                else doc_analyze(dataset, ocr=False)
            )
            pipe = infer.pipe_txt_mode(data_reader=None)

            blocks: list[dict] = []
            for page_index, page_text in enumerate(pipe, start=1):
                text = (page_text or "").strip()
                if len(text) >= config.MIN_PAGE_TEXT_LENGTH:
                    blocks.append({"text": text, "page": page_index, "type": "text"})
            return blocks or None
        except Exception:
            # minerU 的 API 在不同版本间差异较大，失败即视为不可用
            self._mineru_failed = True
            return None

    # ------------------------------------------------------------ PyMuPDF

    def _parse_with_pymupdf(self, pdf_path: Path) -> list[dict]:
        """用 PyMuPDF 逐页抽取文本并剥掉水印层，这是保证可用的兜底方案。"""
        import pymupdf as fitz

        blocks: list[dict] = []
        with fitz.open(pdf_path) as doc:
            for page_index, text in enumerate(_strip_watermark(doc), start=1):
                if len(text) >= config.MIN_PAGE_TEXT_LENGTH:
                    blocks.append({"text": text, "page": page_index, "type": "text"})
        return blocks

    # ------------------------------------------------------------ 表格

    def _extract_tables(self, pdf_path: Path) -> list[dict]:
        """用 pdfplumber 抽取表格，每张表作为一个独立块。"""
        try:
            import pdfplumber
        except ImportError:
            return []

        blocks: list[dict] = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    for table in page.extract_tables() or []:
                        text = _table_to_text(table)
                        if len(text) >= config.MIN_CHUNK_LENGTH:
                            blocks.append({"text": text, "page": page_index, "type": "table"})
        except Exception:
            return []
        return blocks

    # ------------------------------------------------------------ OCR

    def _get_ocr(self):
        """懒加载 OCR 引擎（优先 VL 版面分析），装不上返回 None。"""
        if self._ocr_failed:
            return None
        if self._ocr_engine is None:
            try:
                import ocr_engine

                self._ocr_engine = ocr_engine.get_engine()
            except Exception:
                self._ocr_engine = None
            if self._ocr_engine is None:
                self._ocr_failed = True
        return self._ocr_engine

    def _ocr_images(self, pdf_path: Path) -> list[dict]:
        """把页面里的图片渲染出来做 OCR，识别扫描件和图表中的文字。"""
        if not self.use_ocr:
            return []

        engine = self._get_ocr()
        if engine is None:
            return []

        import pymupdf as fitz

        blocks: list[dict] = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_index, page in enumerate(doc, start=1):
                    for img in page.get_images(full=True):
                        image = _render_image(doc, img)
                        if image is None:
                            continue
                        # 引擎内部已吞掉单张图的异常，坏图不会带崩整份结果
                        result = engine.recognize(image)
                        # 表格排在正文前面，读起来更接近原图的版面顺序
                        parts = [p for p in (result.get("table"), result.get("text")) if p]
                        text = "\n\n".join(parts).strip()
                        if len(text) >= config.MIN_PAGE_TEXT_LENGTH:
                            blocks.append(
                                {"text": text, "page": page_index, "type": "image"}
                            )
        except Exception:
            return []
        return blocks

    # ------------------------------------------------------------ 统一入口

    def parse(self, pdf_path: str | Path) -> list[dict]:
        """解析 PDF，返回按页码排序的块列表。"""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"只支持 PDF 文件，收到：{path.suffix}")

        blocks: list[dict] = []

        if self.use_mineru:
            mineru_blocks = self._parse_with_mineru(path)
            if mineru_blocks:
                blocks = mineru_blocks

        if not blocks:
            blocks = self._parse_with_pymupdf(path)

        if not blocks:
            # 没有任何文本层，说明是扫描件，只能靠 OCR
            blocks = self._ocr_images(path)
            if blocks:
                return blocks

        # 表格作为补充内容，按页码插到同一页的正文之后
        tables = self._extract_tables(path)
        if tables:
            by_page: dict[int, list[dict]] = {}
            for table in tables:
                by_page.setdefault(table["page"], []).append(table)

            merged: list[dict] = []
            for block in blocks:
                merged.append(block)
                merged.extend(by_page.pop(block["page"], []))
            # 落在没有正文的页面上的表格单独补上
            for leftover in by_page.values():
                merged.extend(leftover)
            blocks = merged

        ocr_blocks = self._ocr_images(path)
        if ocr_blocks:
            existing = {(b["page"], b["text"][:50]) for b in blocks}
            blocks.extend(b for b in ocr_blocks if (b["page"], b["text"][:50]) not in existing)

        blocks.sort(key=lambda b: b["page"])
        return blocks


def parse_pdf(pdf_path: str | Path, **kwargs) -> list[dict]:
    """便捷函数：解析单个 PDF。"""
    return PDFParser(**kwargs).parse(pdf_path)


def pdf_to_text(pdf_path: str | Path, **kwargs) -> str:
    """便捷函数：解析并拼成全文，块之间用空行分隔。"""
    return "\n\n".join(b["text"] for b in parse_pdf(pdf_path, **kwargs))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python pdf_parser.py <文件.pdf>")
        raise SystemExit(1)

    for item in parse_pdf(sys.argv[1]):
        preview = item["text"][:60].replace("\n", " ")
        print(f"[第{item['page']}页/{item['type']}] {preview}...")
