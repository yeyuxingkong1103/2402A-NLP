# -*- coding: utf-8 -*-
"""
PDF 解析模块（文字层）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：基于 PyMuPDF 提取 PDF 每页文字，供后续分块与向量化使用。
     表格解析见 table_parser.py（工单03），图像解析见 image_parser.py（工单04）。
"""
import re
import fitz  # PyMuPDF

# 页眉/页脚模式：公司名+文档名行、1-1-XX 编码行、独立数字页码行
_HEADER_PATTERNS = [
    re.compile(r"^武汉兴图新科电子股份有限公司\s+招股意向书\s*$"),
    re.compile(r"^1-1-\d+\s*$"),
    re.compile(r"^\d{1,4}\s*$"),
    re.compile(r"^武汉兴图新科电子股份有限公司\s*$"),
    re.compile(r"^招股意向书\s*$"),
]


def clean_page_text(text):
    """清洗页眉页脚噪音，避免其稀释分块语义（提升向量检索精度）"""
    lines = text.split("\n")
    kept = []
    for ln in lines:
        s = ln.strip()
        if any(p.match(s) for p in _HEADER_PATTERNS):
            continue
        kept.append(ln)
    return "\n".join(kept)


def parse_pdf_text(pdf_path, clean_headers=True):
    """逐页提取 PDF 文字（可选清洗页眉页脚）
    参数 clean_headers: False 时不清洗（仅工单02"优化前"基线对比使用）
    返回: [{"page": 页码(1起), "text": 页面文本}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text and clean_headers:
            text = clean_page_text(text)
        if text:  # 空白页跳过
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def pdf_page_count(pdf_path):
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n
