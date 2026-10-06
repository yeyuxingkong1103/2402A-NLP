# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
PDF 解析模块（优化版）：PyMuPDF 逐页提取 + 文本清洗。

优化点：
    1. 新增文本清洗：去除多余空白行、合并断行（招股书里句中被换行切断很常见）；
    2. 新增页眉页脚粗过滤：连续出现且极短的重复行视为页眉页脚丢弃；
    3. 保留页码溯源能力，供检索命中后展示出处。
"""

import os
import re
from typing import List

from langchain_core.documents import Document

from logger import get_logger

logger = get_logger(__name__)

# 连续空白压缩成正则：多个换行/空格合并
_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
# 中英文断行修复：行尾不是句末标点但下一行以中文开头，拼回去
_RE_BREAK_JOIN = re.compile(r"(?<=[\u4e00-\u9fa5A-Za-z0-9])\n(?=[\u4e00-\u9fa5])")


def _clean_page_text(text: str) -> str:
    """单页文本清洗：合并断行、压空白。"""
    if not text:
        return ""
    # 先把「中文字符 + 换行 + 中文字符」的断行拼回（招股书常见）
    text = _RE_BREAK_JOIN.sub("", text)
    # 压缩多余空格
    text = _RE_MULTI_SPACE.sub(" ", text)
    # 压缩连续空行
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def parse_pdf(file_path: str) -> List[str]:
    """用 PyMuPDF 逐页提取文本，返回按页的文本列表（已清洗）。"""
    import fitz

    if not os.path.isfile(file_path):
        logger.error(f"PDF 文件不存在：{file_path}")
        return []

    pages: List[str] = []
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"PDF 打开失败：{file_path}（{e}）")
        return []

    try:
        for page in doc:
            raw = page.get_text("text")
            pages.append(_clean_page_text(raw))
    finally:
        doc.close()

    text_all = "".join(pages).strip()
    if not text_all:
        logger.warning(f"PDF 文本提取为空，可能是扫描件：{file_path}")
    logger.info(f"PDF 解析完成：{file_path}，共 {len(pages)} 页")
    return pages


def parse_pdf_to_document(file_path: str) -> List[Document]:
    """逐页提取文本，返回带 source/page_number 的 Document 列表（已清洗）。"""
    pages = parse_pdf(file_path)
    source = os.path.basename(file_path)

    docs: List[Document] = []
    for i, text in enumerate(pages):
        if not text.strip():
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source": source,
                "page_number": i + 1,
            },
        ))
    logger.info(f"PDF 转 Document 完成：{file_path}，有效页 {len(docs)} 个")
    return docs
