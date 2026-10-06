# -*- coding: utf-8 -*-
"""
PDF 解析模块 V2 - 语义感知分块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

核心改进 (对比 V1):
    1. 按段落语义切分 (而非固定字符数)
    2. 动态块大小: 短段落合并, 长段落拆分
    3. 表格独立提取为结构化文本块
    4. 标题识别: "第X章"、"一、" 等模式独立成块
"""
import os
import json
import re
import logging
from typing import List, Dict

import config_v2 as config

logger = logging.getLogger(__name__)

# 标题识别正则
HEADER_PATTERNS = [
    r"^第[一二三四五六七八九十百0-9]+章",
    r"^第[一二三四五六七八九十百0-9]+节",
    r"^[一二三四五六七八九十]+、",
    r"^\d+\.\s",
    r"^\d+\.\d+\s",
]


def _is_header(text: str) -> bool:
    """判断一段文字是否为标题"""
    stripped = text.strip()
    if not stripped or len(stripped) > 50:
        return False
    for pat in HEADER_PATTERNS:
        if re.match(pat, stripped):
            return True
    # 短文本且以中文数字开头也视为标题
    if len(stripped) < 20 and re.match(r"^[一二三四五六七八九十]", stripped):
        return True
    return False


def _split_paragraphs(text: str) -> List[str]:
    """
    将文本按语义边界切分为段落列表
    规则:
      - 双换行 → 段落分隔
      - 单换行 + 首行空格 → 段落分隔
      - 否则合并为长段落
    """
    # 先按双换行切
    raw = re.split(r"\n\s*\n", text)
    paragraphs = []
    for r in raw:
        r = r.strip()
        if not r:
            continue
        # 单行文本视为一段
        if "\n" not in r:
            paragraphs.append(r)
            continue
        # 多行文本: 尝试按行内边界切
        lines = [l.strip() for l in r.split("\n") if l.strip()]
        current = ""
        for line in lines:
            # 标题行或新段落开头
            if _is_header(line) or (len(line) > 0 and re.match(r"^[ 　]+", line)):
                if current:
                    paragraphs.append(current)
                current = line
            else:
                current = (current + " " + line).strip()
        if current:
            paragraphs.append(current)
    return paragraphs


def _dynamic_chunk(paragraphs: List[str]) -> List[str]:
    """
    动态处理段落: 短合并, 长拆分

    Returns:
        语义块列表 (每个块长度在 MIN_PARA_LEN ~ MAX_PARA_LEN 之间)
    """
    min_len = config.MIN_PARA_LEN
    max_len = config.MAX_PARA_LEN
    overlap = config.MERGE_OVERLAP

    chunks = []
    buffer = ""

    for para in paragraphs:
        # 长段落直接拆分
        if len(para) > max_len:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            # 长段落按句号/逗号拆分后再合并
            sub_sentences = re.split(r"(?<=[。！？；;])\s*", para)
            sub = ""
            for s in sub_sentences:
                if len(sub) + len(s) <= max_len:
                    sub += s
                else:
                    if sub:
                        chunks.append(sub.strip())
                    sub = s
            if sub:
                chunks.append(sub.strip())
            continue

        # 标题独立成块, 不与后面合并
        if _is_header(para):
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.append(para)
            continue

        # 短段落尝试合并
        if len(buffer) + len(para) + 1 <= max_len:
            buffer = (buffer + " " + para).strip() if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            # overlap 保留上一块尾部
            buffer = (buffer[-overlap:] + " " + para).strip() if len(buffer) > overlap else para

    if buffer:
        chunks.append(buffer)

    return chunks


def extract_text_from_pdf(pdf_path: str = None) -> List[Dict]:
    """从 PDF 提取文本, 按页组织"""
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
            logger.info(f"[V2 解析] 第 {i} 页, 字符 {len(text)}")
    logger.info(f"[V2 解析] 共 {len(pages)} 页")
    return pages


def extract_tables_structured(pdf_path: str = None) -> List[Dict]:
    """
    提取 PDF 表格并转为结构化文本块

    Returns:
        List[{"page": int, "text": str, "type": "table"}]
    """
    pdf_path = pdf_path or config.PDF_PATH
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber")

    table_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue
            for ti, table in enumerate(tables):
                # 转为可读文本
                rows = []
                for row in table:
                    cells = [str(c).strip() if c else "" for c in row]
                    # 过滤全空行
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    table_text = f"[表格 {ti+1}]\n" + "\n".join(rows)
                    table_chunks.append({
                        "page": i,
                        "text": table_text,
                        "type": "table",
                    })
    logger.info(f"[V2 表格] 提取 {len(table_chunks)} 个表格块")
    return table_chunks


def build_semantic_chunks(pages: List[Dict], table_chunks: List[Dict] = None) -> List[Dict]:
    """
    构建语义块 (主流程)

    Returns:
        List[{"id": int, "page": int, "text": str, "type": str}]
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        paragraphs = _split_paragraphs(page["text"])
        sem_chunks = _dynamic_chunk(paragraphs)
        for c in sem_chunks:
            if c.strip():
                chunks.append({
                    "id": chunk_id,
                    "page": page["page"],
                    "text": c.strip(),
                    "type": "text",
                })
                chunk_id += 1

    # 追加表格块
    if table_chunks:
        for tc in table_chunks:
            chunks.append({
                "id": chunk_id,
                "page": tc["page"],
                "text": tc["text"],
                "type": "table",
            })
            chunk_id += 1

    logger.info(f"[V2 分块] 共 {len(chunks)} 个语义块 (含表格)")
    return chunks


def parse_and_save_v2(pdf_path: str = None, output_path: str = None) -> List[Dict]:
    """完整解析流程"""
    output_path = output_path or config.CHUNKS_PATH
    pages = extract_text_from_pdf(pdf_path)
    tables = extract_tables_structured(pdf_path)
    chunks = build_semantic_chunks(pages, tables)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"[V2] 已保存 {len(chunks)} 个语义块到 {output_path}")

    # 统计
    text_count = sum(1 for c in chunks if c["type"] == "text")
    table_count = sum(1 for c in chunks if c["type"] == "table")
    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
    logger.info(f"[V2 统计] 文本块={text_count}, 表格块={table_count}, 平均长度={avg_len:.0f}")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = parse_and_save_v2()
    print(f"V2 分块完成: {len(chunks)} 个")
    if chunks:
        print("示例块:")
        for c in chunks[:3]:
            print(f"  [p{c['page']}, {c['type']}] {c['text'][:80]}")
