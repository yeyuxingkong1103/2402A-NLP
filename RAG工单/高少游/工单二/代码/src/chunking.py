# -*- coding: utf-8 -*-
"""分块模块（优化版）
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

本模块对应优化方案中的【优化点 2：分块优化】。相较基线（固定 500/80 的
无结构递归切片），做了如下增强：

1. 结构感知分块：先按“第X节 / 一、/（一）”等标题把页面正文切成语义小节，
   再在小节内做递归切片，避免跨小节“串味”；
2. 表格原子块：表格整体作为一个不可切分的块（type=table），保证财务数据的
   行列完整，解决基线中“数值被截断 / 表头与数据分离”导致的漏召回；
3. 父子块（Parent-Child）：子块（小块）用于精确检索，父块（小节全文）用于
   提供完整上下文与答案抽取，兼顾“检索精度”与“召回完整性”；
4. 过短块合并、超长块切分，元数据（页码 / 小节标题 / 块类型）齐全，便于溯源。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.pdf_parser import PageDoc, parse_pdf

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百零〇\d]+[章节]\s*[^\n]{0,40}"
    r"|[一二三四五六七八九十]+、[^\n]{0,40}"
    r"|（[一二三四五六七八九十]+）[^\n]{0,40}"
    r"|\d+(?:\.\d+){1,3}\s+[^\n]{0,40})\s*$"
)

_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


@dataclass
class Chunk:
    """一个可检索文本块。"""

    content: str
    page: int
    section: str = ""
    ctype: str = "text"        # text / table
    parent: str = ""           # 父块（小节全文），用于上下文扩展
    index: int = 0


def _split_sections(text: str) -> List[tuple[str, str]]:
    """把页面正文按标题切分为 (小节标题, 小节正文) 列表。"""
    lines = text.splitlines()
    sections: List[tuple[str, str]] = []
    cur_title = ""
    buf: List[str] = []
    for ln in lines:
        if ln.strip() and _HEADING_RE.match(ln.strip()) and len(ln.strip()) <= 45:
            if buf:
                sections.append((cur_title, "\n".join(buf).strip()))
                buf = []
            cur_title = ln.strip()
        else:
            buf.append(ln)
    if buf:
        sections.append((cur_title, "\n".join(buf).strip()))
    return [(t, b) for t, b in sections if b]


def build_chunks(pdf_path=config.PDF_PATH,
                 chunk_size: int = config.CHUNK_SIZE,
                 chunk_overlap: int = config.CHUNK_OVERLAP) -> List[Chunk]:
    """解析 PDF 并生成结构感知的父子块。"""
    pages: List[PageDoc] = parse_pdf(pdf_path, with_tables=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        keep_separator=True,
    )
    chunks: List[Chunk] = []
    idx = 0
    for pg in pages:
        # ---- 1) 表格：整体作为原子块，父块 = 本页表格集合 ----
        # 表格块的章节取本页标题集合，便于“概览/基本情况”类问题的章节先验命中
        table_section = "；".join(pg.headings) if pg.headings else "表格"
        table_parent = "\n\n".join(pg.tables)
        for t in pg.tables:
            chunks.append(Chunk(content=f"[表格]\n{t}", page=pg.page_no,
                                section=table_section, ctype="table",
                                parent=table_parent, index=idx))
            idx += 1

        # ---- 2) 正文：按小节切分后再递归切片 ----
        if not pg.text.strip():
            continue
        for title, body in _split_sections(pg.text):
            parent_text = (f"{title}\n{body}" if title else body).strip()
            if len(parent_text) < config.MIN_CHUNK_SIZE:
                continue
            for piece in splitter.split_text(parent_text):
                piece = piece.strip()
                if len(piece) < 12:
                    continue
                chunks.append(Chunk(content=piece, page=pg.page_no,
                                    section=title or "正文", ctype="text",
                                    parent=parent_text[:2000], index=idx))
                idx += 1
    logger.info("生成块数: %d（表格块 %d）", len(chunks),
                sum(1 for c in chunks if c.ctype == "table"))
    return chunks


def chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    """把 Chunk 列表转换为 LangChain Document（用于向量化）。"""
    return [
        Document(
            page_content=c.content,
            metadata={
                "page": c.page,
                "section": c.section,
                "ctype": c.ctype,
                "parent": c.parent,
                "chunk_index": c.index,
            },
        )
        for c in chunks
    ]


# ---------------- 句子级索引（供抽取式答案合成使用） --------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[。；！？])")


def split_sentences(text: str) -> List[str]:
    """把文本切分为句子列表（保留标点，去噪）。"""
    out: List[str] = []
    for seg in _SENT_SPLIT_RE.split(text or ""):
        s = seg.strip()
        if len(s) >= 8:
            out.append(s)
    return out


@dataclass
class SentenceUnit:
    """句子级检索单元。"""

    text: str
    page: int
    section: str = ""
    parent: str = ""


def build_sentence_index(chunks: List[Chunk]) -> List[SentenceUnit]:
    """基于全部块构建句子级索引（表格块按行拆分为“句子”）。"""
    units: List[SentenceUnit] = []
    for c in chunks:
        if c.ctype == "table":
            for row in c.content.splitlines():
                row = row.strip()
                if row.startswith("|") and not set(row) <= set("|- "):
                    units.append(SentenceUnit(row, c.page, c.section, c.parent))
        else:
            for s in split_sentences(c.content):
                units.append(SentenceUnit(s, c.page, c.section, c.parent))
    return units


# ---------------- 基线分块（复刻 01 工单，用于对比） --------------------------
def build_baseline_chunks(pdf_path=config.PDF_PATH,
                          chunk_size: int = config.BASE_CHUNK_SIZE,
                          chunk_overlap: int = config.BASE_CHUNK_OVERLAP) -> List[Chunk]:
    """复刻 01 工单的原始切片方式：按页取文本，固定长度递归切片，无结构感知。

    仅用于生成“优化前”对照结果，保证 before/after 在同一数据上可比。
    """
    from src.pdf_parser import parse_pdf as _raw_parse

    pages: List[PageDoc] = _raw_parse(pdf_path, with_tables=True, clean_boilerplate=False)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""], keep_separator=True,
    )
    chunks: List[Chunk] = []
    idx = 0
    for pg in pages:
        content = pg.content
        if not content:
            continue
        for piece in splitter.split_text(content):
            chunks.append(Chunk(content=piece, page=pg.page_no,
                                section="", ctype="text", parent="", index=idx))
            idx += 1
    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO)
    cs = build_chunks()
    print("chunks:", len(cs))
    print("table chunks:", sum(1 for c in cs if c.ctype == "table"))
    for c in cs[:3]:
        print("-" * 40)
        print(c.page, c.ctype, c.section[:30])
        print(c.content[:200])