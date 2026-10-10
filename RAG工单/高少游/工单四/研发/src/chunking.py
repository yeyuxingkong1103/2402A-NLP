# -*- coding: utf-8 -*-
"""分块模块（图像内容解析及检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

本模块在 02 工单「结构感知分块 + 表格原子块 + 父子块」基础上，针对表格做
进一步强化（本工单核心）：

1. 表格双块策略：同一张表生成两类原子块——
   - `table` 块：Markdown 结构（保留行列关系，供结构化阅读与答案拼装）；
   - `table_kv` 块：键值对行（“表头：取值”强绑定，供数值/比例型问题精准召回）；
2. 表题入块：把表题（caption）写入块的章节字段，提升“发行概况 / 关联方”等
   章节先验的命中率；
3. 跨页表格以表题为父块：同一表题下的多页表格共享父块，保证跨页数据可整体召回；
4. 多文档支持：每个块携带 source 字段，便于答案溯源到具体招股说明书。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from src import config
from src.pdf_parser import PageDoc, parse_all, parse_pdf

logger = logging.getLogger(__name__)


def _recursive_splitter(**kwargs):
    """惰性构造 RecursiveCharacterTextSplitter。

    性能说明：`langchain_text_splitters` 的包级 `__init__` 会连带导入
    sentence_transformers → transformers → torch（本机实测合计约 250s）。
    而该分割器只在「构建知识库」时需要，检索/问答链路并不使用；故改为函数内惰性导入，
    避免 app 首屏与查询链路被迫承担这一与自身无关的导入开销。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(**kwargs)

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
    ctype: str = "text"        # text / table / table_kv
    parent: str = ""           # 父块（小节全文 / 表题下全部表格），用于上下文扩展
    source: str = ""           # 所属文档文件名
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


def _table_parent(pg: PageDoc, tbl) -> str:
    """表格的父块：本页同表题的全部表格内容（跨页表格共享）。"""
    same = [t for t in pg.tables if t.caption and t.caption == tbl.caption]
    if not same:
        same = [tbl]
    return "\n\n".join(t.text for t in same)[:2000]


def build_chunks(pdf_paths=None,
                 chunk_size: int = config.CHUNK_SIZE,
                 chunk_overlap: int = config.CHUNK_OVERLAP,
                 use_table_parser: bool | None = None) -> List[Chunk]:
    """解析 PDF（多文档）并生成结构感知的父子块。"""
    paths = pdf_paths or config.PDF_PATHS
    use_table_parser = config.USE_TABLE_PARSER if use_table_parser is None else use_table_parser
    pages: List[PageDoc] = []
    for p in paths:
        pages.extend(parse_pdf(p, with_tables=True, use_table_parser=use_table_parser))

    splitter = _recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        keep_separator=True,
    )
    chunks: List[Chunk] = []
    idx = 0
    for pg in pages:
        # ---- 1) 表格：结构化原子块（Markdown + 键值对双块） ----
        for tbl in pg.tables:
            section = tbl.caption or ("；".join(pg.headings) if pg.headings else "表格")
            parent = _table_parent(pg, tbl)
            md = tbl.markdown
            if md:
                chunks.append(Chunk(content=f"[表格]\n{md}", page=pg.page_no,
                                    section=section, ctype="table",
                                    parent=parent, source=pg.source, index=idx))
                idx += 1
            # 键值对块：同一张表的键值对行合并为少量块（每块最多 12 行），
            # 既保留“表头—取值”强绑定，又避免块数量爆炸拖慢向量化。
            kv_lines = tbl.kv_lines
            for j in range(0, len(kv_lines), 12):
                group = kv_lines[j:j + 12]
                chunks.append(Chunk(content="[表格行]\n" + "\n".join(f"[表格行] {k}" for k in group),
                                    page=pg.page_no, section=section, ctype="table_kv",
                                    parent=parent, source=pg.source, index=idx))
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
                                    parent=parent_text[:2000], source=pg.source, index=idx))
                idx += 1
    logger.info("生成块数: %d（表格块 %d / 键值对块 %d）", len(chunks),
                sum(1 for c in chunks if c.ctype == "table"),
                sum(1 for c in chunks if c.ctype == "table_kv"))
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
                "source": c.source,
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


# ---------------- 基线分块（复刻 01/02 工单，用于对比） -----------------------
def build_baseline_chunks(pdf_paths=None,
                          chunk_size: int = config.BASE_CHUNK_SIZE,
                          chunk_overlap: int = config.BASE_CHUNK_OVERLAP) -> List[Chunk]:
    """复刻 01/02 工单的原始切片方式：表格被当作普通文本行，无结构化解析。

    仅用于生成“优化前”对照结果，保证 before/after 在同一数据上可比。
    """
    paths = pdf_paths or config.PDF_PATHS
    pages: List[PageDoc] = []
    for p in paths:
        # use_table_parser=False：退回“表格文本行”的原始形态
        pages.extend(parse_pdf(p, with_tables=True, clean_boilerplate=False,
                               use_table_parser=False))
    splitter = _recursive_splitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""], keep_separator=True,
    )
    chunks: List[Chunk] = []
    idx = 0
    for pg in pages:
        # 基线：用 PyMuPDF 原生表格文本（拍平为文本行），不做行列结构化
        raw_tables = _raw_table_text(pg)
        content = (pg.text + ("\n" + raw_tables if raw_tables else "")).strip()
        if not content:
            continue
        for piece in splitter.split_text(content):
            chunks.append(Chunk(content=piece, page=pg.page_no,
                                section="", ctype="text", parent="",
                                source=pg.source, index=idx))
            idx += 1
    return chunks


def _raw_table_text(pg: PageDoc) -> str:
    """基线用：把结构化表格“退化为”拍平的文本行（模拟 01/02 工单的解析结果）。"""
    lines: List[str] = []
    for tbl in pg.tables:
        for row in tbl.rows:
            cells = [c for c in row if c]
            if cells:
                lines.append(" ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO)
    cs = build_chunks()
    print("chunks:", len(cs))
    for t in ("text", "table", "table_kv"):
        print(f"  {t}: {sum(1 for c in cs if c.ctype == t)}")
    for c in cs[:2]:
        print("-" * 40)
        print(c.source, c.page, c.ctype, c.section[:30])
        print(c.content[:200])