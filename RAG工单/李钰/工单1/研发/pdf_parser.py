# -*- coding: utf-8 -*-
"""
PDF 解析模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能:
    1. 解析《招股说明书1.pdf》提取文字内容
    2. 提取表格数据
    3. 按字符数切块 (chunk) 以便后续向量化
"""
import os
import json
import logging
from typing import List, Dict

import config

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str = None) -> List[Dict]:
    """
    从 PDF 中提取文本, 返回按页组织的文本块列表

    Args:
        pdf_path: PDF 文件路径, 默认使用配置中的路径

    Returns:
        List[{"page": int, "text": str}]
    """
    pdf_path = pdf_path or config.PDF_PATH
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
            logger.info(f"已解析第 {i} 页, 字符数 {len(text)}")
    logger.info(f"PDF 解析完成, 共 {len(pages)} 页")
    return pages


def extract_tables_from_pdf(pdf_path: str = None) -> List[Dict]:
    """
    从 PDF 中提取表格数据

    Returns:
        List[{"page": int, "tables": List[List[str]]}]
    """
    pdf_path = pdf_path or config.PDF_PATH
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber")

    tables_result = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if tables:
                tables_result.append({"page": i, "tables": tables})
    return tables_result


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    将长文本切分为带重叠的文本块

    Args:
        text: 输入文本
        chunk_size: 块大小
        overlap: 重叠字符数

    Returns:
        List[str] 文本块列表
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    """
    将页面文本拼接并切分为 chunks

    Returns:
        List[{"id": int, "page": int, "text": str}]
    """
    chunks = []
    chunk_id = 0
    for page in pages:
        page_text = page["text"]
        if not page_text.strip():
            continue
        page_chunks = chunk_text(page_text)
        for c in page_chunks:
            chunks.append({
                "id": chunk_id,
                "page": page["page"],
                "text": c.strip(),
            })
            chunk_id += 1
    return chunks


def parse_and_save(pdf_path: str = None, output_path: str = None) -> List[Dict]:
    """
    解析 PDF 并将 chunks 保存到 JSON 文件

    Returns:
        List[Dict] chunks 列表
    """
    output_path = output_path or config.CHUNKS_PATH
    pages = extract_text_from_pdf(pdf_path)
    chunks = build_chunks(pages)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存 {len(chunks)} 个文本块到 {output_path}")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = parse_and_save()
    print(f"解析完成, 共生成 {len(chunks)} 个文本块")
    if chunks:
        print("第一个文本块示例:")
        print(chunks[0]["text"][:200])
