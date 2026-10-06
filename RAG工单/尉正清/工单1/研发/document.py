# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""文档处理：PDF 解析（文字 + 表格）+ 文本分块"""
import re
from collections import Counter

import pdfplumber
import pymupdf

from config import CHUNK_SIZE, CHUNK_OVERLAP

# 纯页码样式的行，如 "1-1-62"、"12"、"– 8 –"
_PAGE_NO = re.compile(r"^[\s\-–—_0-9]{1,12}$")


# ---------------- PDF 解析 ----------------
def parse_pdf_text(pdf_path):
    """逐页取文字并去掉页眉页脚。

    招股书每页页眉都是「公司名 + 招股意向书」。实测 1396 个文字块里有 548 个
    （39%）以公司名开头，而工单的 10 个问题全都含公司名 —— 等于近四成的块都
    吃到同等匹配加分，检索排序被拉平，正确答案会掉到第 3~4 位。所以必须先去掉。
    """
    doc = pymupdf.open(pdf_path)
    try:
        pages = [{"page": i, "text": page.get_text("text")}
                 for i, page in enumerate(doc, start=1)]
    finally:
        doc.close()
    return _strip_repeats([p for p in pages if p["text"].strip()])


def _strip_repeats(pages, min_ratio=0.3, min_pages=5):
    """删掉在多页重复出现的行（页眉/页脚）和纯页码行。"""
    if not pages:
        return pages

    counter = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page["text"].splitlines()}:
            if line:
                counter[line] += 1

    threshold = max(min_pages, int(len(pages) * min_ratio))
    repeated = {ln for ln, n in counter.items() if n >= threshold}

    for page in pages:
        page["text"] = "\n".join(
            ln for ln in page["text"].splitlines()
            if ln.strip() and ln.strip() not in repeated and not _PAGE_NO.match(ln)
        )
    return pages


def parse_pdf_tables(pdf_path):
    """抽取表格并转成 Markdown。"""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                md = _table_to_markdown(table)
                if md:
                    tables.append({"page": i, "table": md})
    return tables


def _table_to_markdown(table):
    if not table or not table[0]:
        return ""
    rows = ["| " + " | ".join(_cell(c) for c in row) + " |" for row in table]
    width = max(len(row) for row in table)          # 按最宽的一行定列数，防止错位
    sep = "| " + " | ".join(["---"] * width) + " |"
    return "\n".join([rows[0], sep] + rows[1:])


def _cell(value):
    return "" if value is None else str(value).replace("\n", " ").strip()


# ---------------- 文本分块 ----------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """固定长度滑窗切分，尽量在换行处断开，避免把财务数字切成两半。"""
    text = text.strip()
    if not text:
        return []
    overlap = min(overlap, chunk_size - 1)          # 防止步长 <=0 造成死循环
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = text.rfind("\n", start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
    return chunks


def build_chunks(text_pages, table_blocks):
    """把文字页和表格切成带页码、带类型的块。"""
    chunks = [
        {"text": piece, "page": page["page"], "type": "text"}
        for page in text_pages for piece in split_text(page["text"])
    ]
    chunks += [
        {"text": t["table"], "page": t["page"], "type": "table"}
        for t in table_blocks
    ]
    return chunks


def load_pdf(pdf_path):
    """返回 (文字页列表, 表格块列表)"""
    return parse_pdf_text(pdf_path), parse_pdf_tables(pdf_path)
