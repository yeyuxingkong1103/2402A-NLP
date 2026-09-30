"""PDF 解析与 OCR：正文/图片用 PyMuPDF (fitz)，表格用 pdfplumber，OCR 用 EasyOCR。

输出统一为 list[{"page": int, "text": str}]。这里做基础抽取、去水印、页眉页脚过滤、
表格 Markdown 化与图片 OCR，深度空白归一化交给 ingest.normalize_whitespace。
依赖方向：ingest -> parser（单向）。
"""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def _filter_header_footer(
    lines_per_page: list[list[str]],
    header_size: int = 3,
    footer_size: int = 3,
    min_repeat: int = 2,
) -> list[list[str]]:
    """丢弃重复出现的页眉/页脚行：某行（strip 后）在 >= min_repeat 页出现即判定为页眉/页脚。

    只检查每页前 header_size 行与后 footer_size 行，中间正文行不受影响。
    """
    # 1. 收集每页首尾候选行的归一化文本
    candidates: list[list[str]] = []
    for lines in lines_per_page:
        head = lines[:header_size]
        foot = lines[len(lines) - footer_size:] if footer_size > 0 else []
        candidates.append([ln.strip() for ln in head + foot if ln.strip()])

    # 2. 统计每个候选行出现的页数（同一页内去重）
    page_count: dict[str, int] = {}
    for cands in candidates:
        for ln in set(cands):
            page_count[ln] = page_count.get(ln, 0) + 1
    repeated = {ln for ln, c in page_count.items() if c >= min_repeat}

    # 3. 从每页首尾候选位置剔除重复行
    cleaned: list[list[str]] = []
    for lines in lines_per_page:
        if len(lines) <= header_size + footer_size:
            cleaned.append([ln for ln in lines if ln.strip() not in repeated])
            continue
        head = [ln for ln in lines[:header_size] if ln.strip() not in repeated]
        body = lines[header_size: len(lines) - footer_size]
        foot = [ln for ln in lines[len(lines) - footer_size:] if ln.strip() not in repeated]
        cleaned.append(head + body + foot)
    return cleaned


def _extract_lines_with_y(doc) -> list[list[tuple[int, str]]]:
    """每页用 get_text("dict") 提取文本行，返回 [(y0, text)]，保持阅读顺序、空行丢弃。"""
    pages: list[list[tuple[int, str]]] = []
    for page in doc:
        lines: list[tuple[int, str]] = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if not text:
                    continue
                y = round(line["bbox"][1])
                lines.append((y, text))
        pages.append(lines)
    return pages


def _remove_watermark(
    lines_per_page: list[list[tuple[int, str]]],
    min_pages: int = 3,
    max_len: int = 30,
) -> tuple[list[list[tuple[int, str]]], int]:
    """去掉跨页重复出现的短文本（水印）：某行在 >= min_pages 页出现且长度 < max_len。

    不依赖具体内容，可识别 pymupdf 缺字体导致的乱码（如 "···· ····"）。
    返回 (去水印后的行, 去掉的总行数)。
    """
    page_count: dict[str, int] = {}
    for lines in lines_per_page:
        for text in {t for _, t in lines}:
            page_count[text] = page_count.get(text, 0) + 1
    watermarks = {t for t, c in page_count.items() if c >= min_pages and len(t) < max_len}

    cleaned: list[list[tuple[int, str]]] = []
    removed = 0
    for lines in lines_per_page:
        kept = [(y, t) for y, t in lines if t not in watermarks]
        removed += len(lines) - len(kept)
        cleaned.append(kept)
    return cleaned, removed


def extract_text(pdf_path: str) -> list[dict]:#PyMuPDF 提取正文 + 跨页统计去水印
    """用 PyMuPDF 逐页抽取正文，去水印、过滤页眉页脚，返回 [{"page":int,"text":str}]。

    页码从 1 起，空页跳过（页码不重排）。
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        lines_per_page = _extract_lines_with_y(doc)
    finally:
        doc.close()

    lines_per_page, removed = _remove_watermark(lines_per_page)
    if removed:
        logger.info("去掉了 %d 行水印", removed)

    text_lines_per_page = [[text for _, text in lines] for lines in lines_per_page]
    text_lines_per_page = _filter_header_footer(text_lines_per_page)

    result = []
    for page_no, lines in enumerate(text_lines_per_page, start=1):
        text = "\n".join(ln.rstrip() for ln in lines).strip()
        if text:
            result.append({"page": page_no, "text": text})
    return result


def _header_cell(cell) -> str:
    """表头单元格合并成单行：按换行拆片、各片 strip 后空串拼接。"""
    if cell is None:
        return ""
    return "".join(part.strip() for part in str(cell).splitlines())


def _data_cell(cell) -> str:
    """数据单元格：None 转空串，换行转空格。"""
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").strip()


def _render_table_markdown(table: list[list]) -> str | None:
    """把表格渲染为 Markdown；空表 / 单行表返回 None。

    第一行视为表头，表头单元格合并换行；数据单元格换行转空格。
    """
    if len(table) <= 1:
        return None
    if not any(_data_cell(c) for row in table for c in row):
        return None

    ncols = len(table[0])
    header_cells = [_header_cell(c).replace("|", "\\|") for c in table[0]]
    rows = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for row in table[1:]:
        cells = [_data_cell(c).replace("|", "\\|") for c in row[:ncols]]
        cells += [""] * (ncols - len(cells))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def extract_tables(pdf_path: str) -> list[dict]:#pdfplumber 提取表格 → Markdown
    """用 pdfplumber 提取表格，渲染为 Markdown，每个表格作为独立 chunk。

    空表 / 单行表跳过；返回 [{"page":int,"text":str}]，text 带 "【表格N，第X页】" 标记。
    """
    import pdfplumber

    result = []
    n = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                md = _render_table_markdown(table)
                if md is None:
                    continue
                n += 1
                result.append({"page": page_no, "text": f"【表格{n}，第{page_no}页】\n{md}"})
    logger.info("提取了 %d 个表格", n)
    return result


@lru_cache(maxsize=1)
def _ocr_reader():
    """EasyOCR Reader 单例：模型加载慢，进程内只初始化一次。"""
    import easyocr

    logger.info("初始化 EasyOCR ...")
    return easyocr.Reader(["ch_sim", "en"], gpu=False)


def extract_image_text(image_path: str) -> str:#EasyOCR 识别图片文字
    """用 EasyOCR 识别图片文字，返回换行拼接的文本；失败返回空串并告警。"""
    try:
        lines = _ocr_reader().readtext(image_path, detail=0)
    except Exception:
        logger.warning("OCR 识别失败: %s", image_path, exc_info=True)
        return ""
    return "\n".join(t.strip() for t in lines if t.strip())


def extract_from_image(image_path: str) -> list[dict]:#单张图片走 OCR 入库
    """识别单张图片，返回 [{"page":1,"text":...}]（与 extract_text 契约一致）；无文字返回空列表。"""
    text = extract_image_text(image_path)
    if not text:
        return []
    return [{"page": 1, "text": text}]


def _ocr_image_bytes(raw: bytes, ext: str) -> str:
    """把内嵌图字节写入临时文件后走 extract_image_text 识别，返回文本。"""
    import os
    import tempfile

    suffix = f".{ext}" if ext else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        tmp_path = f.name
    try:
        return extract_image_text(tmp_path)
    finally:
        os.unlink(tmp_path)


def extract_images(pdf_path: str) -> list[dict]:#提取 PDF 内嵌图片
    """提取 PDF 内嵌图片：OCR 出文字则输出 "【第X页图片】"，否则输出尺寸占位。"""
    import pymupdf

    result = []
    seen: set[int] = set()
    doc = pymupdf.open(pdf_path)
    try:
        for page_no, page in enumerate(doc, start=1):
            for xref, _smask, w, h, *_ in page.get_images(full=True):
                if xref in seen:
                    continue
                seen.add(xref)
                info = doc.extract_image(xref)
                text = _ocr_image_bytes(info["image"], info.get("ext", ""))
                if text:
                    result.append({"page": page_no, "text": f"【第{page_no}页图片】\n{text}"})
                else:
                    result.append({"page": page_no, "text": f"【第{page_no}页包含图片1张，尺寸 {w}x{h}】"})
    finally:
        doc.close()
    return result
"""
负责 PDF 文本、表格、内嵌图片以及外部图片的文字提取，分为三部分。第一，`extract_text()`是 PDF 正文提取入口，调用`_extract_lines_with_y()`，基于 PyMuPDF 拿到带 y 坐标的文本行；再调用`_remove_watermark()`，识别跨页重复的短文本作为水印并删除，返回去掉水印后的行和删除行数；接着把坐标信息丢掉，只保留文本，传给`_filter_header_footer()`，统计每页开头和结尾重复出现的行，自动剔除页眉页脚；最后把清理后的文本按页组装成`{"page":页码,"text":页面文本}`结构，空页面直接跳过。第二，`extract_tables()`用来提取 PDF 里的表格，使用 pdfplumber 读取表格数据，调用`_render_table_markdown()`，`_header_cell()`处理表头单元格、`_data_cell()`处理数据单元格，把表格转成 Markdown 格式；空表或者只有一行的表格直接跳过，成功提取的表格会加上【表格 N，第 X 页】标记，作为独立条目返回。第三，图片 OCR 相关能力。`extract_from_image()`是外部图片的入口，调用`extract_image_text()`，`extract_image_text()`使用`_ocr_reader()`，用`@lru_cache`做 EasyOCR 单例，模型只加载一次，识别图片文字；`extract_images()`提取 PDF 内部嵌入图片，拿到图片二进制字节，调用`_ocr_image_bytes()`生成临时文件做 OCR 识别。识别出文字就带上图片标记保存；识别不出文字，就记录图片尺寸占位。OCR 识别出错时只打告警，返回空字符串，不中断整体解析流程。
"""