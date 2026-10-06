# -*- coding: utf-8 -*-
# 【表格感知分块模块 · chunker.py】整表块 + 表格行陈述句块 + 正文标题感知打包
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""分块层：把解析层 Block 切分为携带标题路径、页码与公司主体的检索块。

三类检索块（对应设计文档第 4 章）：
- text：正文按句读切句后贪心打包，块间保留字符重叠；
- table：整张清洗后 Markdown 表独立成块（父文本=整表）；
- table_row：表格每一数据行的自然语言陈述句独立成块（Small-to-Big：
  行块负责高信号召回，父文本返回整张表，答案可溯源到行与页码）。
"""
import re
from dataclasses import dataclass, field
from typing import List

from config import CONFIG
from pdf_parser import Block

_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])|(?<=\.)\s+")


@dataclass
class Chunk:
    """检索块：子块文本用于检索，父文本用于展示作答。"""

    chunk_id: int
    text: str
    parent_text: str
    page_no: int
    heading_path: str
    company: str = ""
    chunk_type: str = "text"   # text / table / table_row
    table_title: str = ""
    row_claim: str = ""        # 表格行块对应的陈述句（便于答案抽取）


def split_sentences(text: str) -> List[str]:
    """按中文/英文句读切句，过短句子并入相邻句。

    :param text: 段落原文
    :return: 句子列表
    """
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    merged: List[str] = []
    for part in parts:
        if merged and len(part) < 8:
            merged[-1] += part
        else:
            merged.append(part)
    return merged


def _pack(sentences: List[str], size: int, overlap: int) -> List[str]:
    """贪心打包句子为目标长度块，块间按字符滑窗重叠。

    :param sentences: 句子列表
    :param size: 目标块字符数
    :param overlap: 重叠字符数
    :return: 块文本列表
    """
    packs: List[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) > size:
            packs.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = tail + sent
        else:
            buf += sent
    if buf.strip():
        packs.append(buf.strip())
    return packs


def build_chunks(blocks: List[Block]) -> List[Chunk]:
    """把结构化 Block 列表构建为表格感知检索 Chunk 列表。

    :param blocks: pdf_parser 输出的 Block 列表（可来自多本 PDF）
    :return: 带编号、页码、标题路径、公司主体与块类型的 Chunk 列表
    """
    chunks: List[Chunk] = []
    cid = 0

    # 同标题路径+同页的正文聚合为“父段落”，供 Small-to-Big 返回
    parents: dict = {}
    for b in blocks:
        if b.type == "text":
            parents.setdefault((b.company, b.heading_path, b.page_no),
                               []).append(b.text)

    for b in blocks:
        head = f"【{b.heading_path}】" if b.heading_path else ""
        ent = f"主体：{b.company}\n" if b.company else ""

        if b.type == "table":
            caption = b.meta.get("caption", b.table_title)
            title_line = f"表格标题：{caption}\n" if caption else ""
            # ① 整表块
            cid += 1
            whole_text = f"{head}{ent}{title_line}{b.text}"
            chunks.append(Chunk(
                chunk_id=cid, text=whole_text, parent_text=b.text,
                page_no=b.page_no, heading_path=b.heading_path,
                company=b.company, chunk_type="table",
                table_title=caption))
            # ② 逐行陈述句块（高信号、易命中、答案可定位到单元格）
            for claim in b.meta.get("claims", []):
                cid += 1
                row_text = f"{head}{ent}{title_line}{claim}"
                chunks.append(Chunk(
                    chunk_id=cid, text=row_text, parent_text=b.text,
                    page_no=b.page_no, heading_path=b.heading_path,
                    company=b.company, chunk_type="table_row",
                    table_title=caption, row_claim=claim))
            continue

        sentences = split_sentences(b.text)
        parent = "".join(parents.get(
            (b.company, b.heading_path, b.page_no), [b.text]))
        for pack in _pack(sentences, CONFIG.chunk_size, CONFIG.chunk_overlap):
            cid += 1
            chunks.append(Chunk(
                chunk_id=cid, text=f"{head}{ent}{pack}",
                parent_text=parent if len(parent) <= 1500 else parent[:1500],
                page_no=b.page_no, heading_path=b.heading_path,
                company=b.company, chunk_type="text"))
    return chunks


def build_naive_chunks(blocks: List[Block], size: int = 450,
                       overlap: int = 80) -> List[Chunk]:
    """基线分块：忽略表格结构，正文按定长窗口滑窗切分（模拟工单 01 朴素链路）。

    表格块在基线下被拍平为单元格顺序文本，行列归属丢失，用于优化前后对比。

    :param blocks: pdf_parser 输出 Block（表格为 Markdown 文本）
    :param size: 定长窗口
    :param overlap: 重叠长度
    :return: Chunk 列表（全部为 text 类型）
    """
    full_pages: dict = {}
    for b in blocks:
        full_pages.setdefault((b.company, b.page_no), []).append(
            re.sub(r"[|\-\n]+", " ", b.text))
    chunks: List[Chunk] = []
    cid = 0
    for (company, page_no), segs in sorted(full_pages.items(),
                                           key=lambda x: x[0][1]):
        text = re.sub(r"\s+", " ", "".join(segs)).strip()
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                cid += 1
                chunks.append(Chunk(
                    chunk_id=cid, text=f"主体：{company}\n{piece}",
                    parent_text=piece, page_no=page_no, heading_path="",
                    company=company, chunk_type="text"))
            start += max(size - overlap, 1)
    return chunks
