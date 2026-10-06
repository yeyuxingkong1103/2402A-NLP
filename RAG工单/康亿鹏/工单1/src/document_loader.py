# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：PDF 解析模块。负责从 PDF 中提取正文文字与表格内容，
          并转换为 LangChain 的 Document 对象，供后续切分与向量化使用。
"""
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

import config
from src.utils import clean_text, logger


class DocumentParseError(Exception):
    """PDF 解析异常（文件不存在、加密、损坏等）。"""


def _block_in_margin(block_bbox: tuple, page_height: float, margin_ratio: float) -> bool:
    """判断文本块是否落在页眉/页脚区域（用于剔除页眉、页脚与页码）。"""
    _, y0, _, y1 = block_bbox
    top = page_height * margin_ratio
    bottom = page_height * (1 - margin_ratio)
    return y1 <= top or y0 >= bottom


def _extract_page_text(page, margin_ratio: float) -> str:
    """提取单页正文（剔除页眉页脚），按布局顺序拼接。"""
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
    return clean_text("\n".join(parts))


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

    documents = []
    for page in doc:
        page_no = page.number + 1
        text = _extract_page_text(page, margin_ratio)
        if text:
            documents.append(
                Document(page_content=text, metadata={"source": path.name, "page": page_no, "type": "text"})
            )
        if enable_tables:
            for md in _extract_page_tables(page):
                documents.append(
                    Document(page_content=md, metadata={"source": path.name, "page": page_no, "type": "table"})
                )
    doc.close()

    if not documents:
        raise DocumentParseError(f"未能从 {path.name} 中提取到任何文本，请确认该 PDF 是否包含文字层。")

    logger.info("PDF 解析完成：%s，共 %d 页，生成 %d 个文档片段", path.name, len(documents), len(documents))
    return documents
