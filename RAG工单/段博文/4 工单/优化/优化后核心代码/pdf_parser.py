# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""
PDF 解析模块（图像内容解析增强版）：文本提取 + 表格识别 + 图像语义解析。

核心功能：
    1. 文本提取：PyMuPDF 提取普通文本块；
    2. 表格识别：pdfplumber 识别表格并转 Markdown；
    3. 图像语义：多模态模块定位结构图/统计图（含矢量图），OCR + LLM 生成语义块；
    4. 混合输出：文本/表格/图像三类块统一为 Document 对象。
"""

import os
from typing import List

import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document

from config import (
    TABLE_EXTRACTION_ENABLED,
    TABLE_TO_MARKDOWN,
    TABLE_MIN_ROWS,
    IMAGE_EXTRACTION_ENABLED,
)
from image_parser import extract_image_documents
from logger import get_logger

logger = get_logger(__name__)


class TableBlock:
    """表格块数据结构。"""

    def __init__(
        self,
        page_number: int,
        table_index: int,
        headers: List[str],
        rows: List[List[str]],
        raw_text: str,
    ):
        self.page_number = page_number
        self.table_index = table_index
        self.headers = headers
        self.rows = rows
        self.raw_text = raw_text
        self.table_type = "table"


def parse_pdf_to_document(pdf_path: str) -> List[Document]:
    """解析 PDF 文件为 Document 列表，包含文本和表格。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        List[Document]: 包含文本块和表格块的 Document 列表
    """
    if not os.path.isfile(pdf_path):
        logger.error(f"PDF 文件不存在：{pdf_path}")
        return []

    documents = []
    file_name = os.path.basename(pdf_path)

    # 1. 先用 PyMuPDF 提取普通文本
    text_docs = _extract_text_blocks(pdf_path, file_name)
    documents.extend(text_docs)
    logger.info(f"提取文本块：{len(text_docs)} 个")

    # 2. 再用 pdfplumber 提取表格
    if TABLE_EXTRACTION_ENABLED:
        table_docs = _extract_table_blocks(pdf_path, file_name)
        documents.extend(table_docs)
        logger.info(f"提取表格块：{len(table_docs)} 个")

    # 3. 多模态图像语义解析（结构图/统计图，含矢量图）
    if IMAGE_EXTRACTION_ENABLED:
        image_docs = extract_image_documents(pdf_path, file_name)
        documents.extend(image_docs)
        logger.info(f"提取图像块：{len(image_docs)} 个")

    # 4. 按页码排序
    documents.sort(key=lambda x: (x.metadata.get("page_number", 0), x.metadata.get("block_index", 0)))

    logger.info(f"PDF 解析完成：{file_name}，共 {len(documents)} 个块")
    return documents


def _extract_text_blocks(pdf_path: str, file_name: str) -> List[Document]:
    """使用 PyMuPDF 提取文本块。"""
    documents = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue

        # 简单分块：按段落分割
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, para in enumerate(paragraphs):
            if len(para) < 20:  # 跳过过短的段落
                continue

            doc_obj = Document(
                page_content=para,
                metadata={
                    "source": file_name,
                    "page_number": page_num,
                    "block_index": idx,
                    "block_type": "text",
                    "chunk_index": -1,  # 由后续切分模块填充
                },
            )
            documents.append(doc_obj)

    doc.close()
    return documents


def _extract_table_blocks(pdf_path: str, file_name: str) -> List[Document]:
    """使用 pdfplumber 提取表格块。"""
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table_idx, table in enumerate(tables):
                if not table or len(table) < TABLE_MIN_ROWS:
                    continue

                # 解析表头和数据行
                headers = [str(cell) if cell else "" for cell in table[0]]
                rows = []
                for row in table[1:]:
                    row_data = [str(cell) if cell else "" for cell in row]
                    if any(cell.strip() for cell in row_data):  # 跳过全空行
                        rows.append(row_data)

                if not rows:
                    continue

                # 转换为 Markdown 格式
                if TABLE_TO_MARKDOWN:
                    markdown_table = _table_to_markdown(headers, rows)
                    content = markdown_table
                else:
                    content = _table_to_text(headers, rows)

                doc_obj = Document(
                    page_content=content,
                    metadata={
                        "source": file_name,
                        "page_number": page_num,
                        "block_index": table_idx,
                        "block_type": "table",
                        "table_headers": headers,
                        "table_rows": len(rows),
                        "chunk_index": -1,
                    },
                )
                documents.append(doc_obj)

    return documents


def _table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """将表格转换为 Markdown 格式。"""
    lines = []

    # 表头
    header_line = "| " + " | ".join(headers) + " |"
    lines.append(header_line)

    # 分隔行
    separator = "|" + "|".join([" --- " for _ in headers]) + "|"
    lines.append(separator)

    # 数据行
    for row in rows:
        row_line = "| " + " | ".join(row) + " |"
        lines.append(row_line)

    return "\n".join(lines)


def _table_to_text(headers: List[str], rows: List[List[str]]) -> str:
    """将表格转换为纯文本格式（备用方案）。"""
    lines = []

    # 表头
    lines.append("表头: " + " | ".join(headers))

    # 数据行
    for row in rows:
        lines.append(" | ".join(row))

    return "\n".join(lines)


def get_pdf_metadata(pdf_path: str) -> dict:
    """获取 PDF 文件元数据。"""
    if not os.path.isfile(pdf_path):
        return {}

    doc = fitz.open(pdf_path)
    metadata = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "page_count": len(doc),
        "file_size": os.path.getsize(pdf_path),
    }
    doc.close()

    return metadata
