# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：PDF 解析模块。负责从 PDF 中提取正文文字与表格内容，
          并转换为 LangChain 的 Document 对象，供后续切分与向量化使用。
"""
import re
from collections import Counter
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

import config
from src.utils import clean_text, logger


class DocumentParseError(Exception):
    """PDF 解析异常（文件不存在、加密、损坏等）。"""


# 工单02 解析优化：识别章节标题，招股说明书常用“第X节 / 第X章 / 一、”等形式
_HEADING_RE = re.compile(r"^(第[一二三四五六七八九十百]+[节章]|[一二三四五六七八九十]{1,3}、)\s*\S")
# 招股说明书的页内编号（如 1-1-3）：逐页不同但属噪声
_PAGE_NO_RE = re.compile(r"^\d{1,3}(-\d{1,3}){1,3}$")


def _is_heading(text: str) -> bool:
    """判断文本块是否为章节标题（用于给片段注入章节上下文）。"""
    t = text.strip()
    if not t or len(t) > 45:
        return False
    if t.endswith(("。", "；", "，")):
        return False
    return bool(_HEADING_RE.match(t))


def _norm_line(text: str) -> str:
    """规范化空白，便于跨页比较同一行文本。"""
    return re.sub(r"\s+", " ", text).strip()


def _find_repeated_lines(page_blocks: list, ratio: float) -> set:
    """找出在超过 ratio 比例页面上重复出现的短行（页眉 / 页脚 / 水印噪声）。

    工单02 解析优化：这类噪声即使不在页边距内，也会出现在几乎每个片段中，
    稀释片段的向量语义、降低检索精度，故统一剔除。
    """
    total = len(page_blocks)
    if total == 0 or ratio <= 0:
        return set()
    counter = Counter()
    for _, blocks in page_blocks:
        lines = {_norm_line(ln) for b in blocks for ln in b.split("\n") if _norm_line(ln)}
        for line in lines:
            counter[line] += 1
    threshold = max(3, int(total * ratio))
    return {line for line, count in counter.items() if count >= threshold and len(line) <= 60}


def _strip_noise(blocks: list, noise: set) -> list:
    """按行剔除噪声：重复出现的页眉/页脚/水印行，以及页内编号行（如 1-1-3）。"""
    cleaned = []
    for block in blocks:
        lines = [
            ln for ln in block.split("\n")
            if _norm_line(ln)
            and _norm_line(ln) not in noise
            and not _PAGE_NO_RE.match(_norm_line(ln))
        ]
        if lines:
            cleaned.append("\n".join(lines))
    return cleaned


def _block_in_margin(block_bbox: tuple, page_height: float, margin_ratio: float) -> bool:
    """判断文本块是否落在页眉/页脚区域（用于剔除页眉、页脚与页码）。"""
    _, y0, _, y1 = block_bbox
    top = page_height * margin_ratio
    bottom = page_height * (1 - margin_ratio)
    return y1 <= top or y0 >= bottom


def _extract_page_blocks(page, margin_ratio: float) -> list:
    """提取单页正文文本块（剔除页眉页脚），保持版面顺序。"""
    height = page.rect.height
    parts = []
    for block in page.get_text("blocks"):
        # block: (x0, y0, x1, y1, text, block_no, block_type)
        if len(block) < 5:
            continue
        text = block[4]
        if not text or not text.strip():
            continue
        if _block_in_margin(block[:4], height, margin_ratio):
            continue
        parts.append(text)
    return parts


def _extract_page_tables(page) -> list:
    """提取单页中的表格，返回 Markdown 文本列表。"""
    tables_md = []
    try:
        tables = page.find_tables()
    except Exception as exc:  # 个别页面的表格识别失败不影响整体解析
        logger.warning("第 %s 页表格解析失败：%s", page.number + 1, exc)
        return tables_md
    for tbl in tables:
        try:
            md = tbl.to_markdown()
        except Exception:
            md = None
        if md and md.strip():
            tables_md.append(md.strip())
    return tables_md


def load_pdf(pdf_path: str, enable_tables: bool = None, margin_ratio: float = None) -> list:
    """解析 PDF 文件，返回 Document 列表。

    每个 Document 的 metadata 包含：
        source  : 文件名
        page    : 页码（从 1 开始）
        type    : text / table
        section : 所属章节标题（工单02 优化，用于分块时注入上下文）
    """
    enable_tables = config.ENABLE_TABLE_PARSING if enable_tables is None else enable_tables
    margin_ratio = config.PAGE_MARGIN_RATIO if margin_ratio is None else margin_ratio

    path = Path(pdf_path)
    if not path.exists():
        raise DocumentParseError(f"PDF 文件不存在：{pdf_path}")

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise DocumentParseError(f"PDF 打开失败：{path.name}（{exc}）") from exc

    if doc.is_encrypted and not doc.authenticate(""):
        raise DocumentParseError(f"PDF 已加密，无法解析：{path.name}")

    # 单次遍历收集全部页面的文本块与表格
    page_blocks = []
    page_tables = []
    for page in doc:
        page_blocks.append((page.number + 1, _extract_page_blocks(page, margin_ratio)))
        page_tables.append(_extract_page_tables(page) if enable_tables else [])
    page_count = doc.page_count
    doc.close()

    # 工单02 解析优化：剔除跨页重复出现的页眉 / 页脚 / 水印噪声（ratio<=0 时整体关闭）
    noise_on = config.REPEATED_BLOCK_RATIO > 0
    noise = _find_repeated_lines(page_blocks, config.REPEATED_BLOCK_RATIO) if noise_on else set()

    documents = []
    current_section = ""
    for idx, (page_no, blocks) in enumerate(page_blocks):
        if noise_on:
            blocks = _strip_noise(blocks, noise)
        # 工单02 解析优化：记录本页所属章节（取本页最后一个标题作为后续片段的章节）
        for block in blocks:
            if _is_heading(block):
                current_section = block.strip()
        text = clean_text("\n".join(blocks))
        if text:
            documents.append(
                Document(page_content=text, metadata={
                    "source": path.name, "page": page_no, "type": "text", "section": current_section})
            )
        for md in page_tables[idx]:
            documents.append(
                Document(page_content=md, metadata={
                    "source": path.name, "page": page_no, "type": "table", "section": current_section})
            )

    if not documents:
        raise DocumentParseError(f"未能从 {path.name} 中提取到任何文本，请确认该 PDF 是否包含文字层。")

    text_count = sum(1 for d in documents if d.metadata.get("type") == "text")
    table_count = len(documents) - text_count
    logger.info("PDF 解析完成：%s，共 %d 页，生成 %d 个文档片段（正文 %d / 表格 %d）",
                path.name, page_count, len(documents), text_count, table_count)
    return documents
