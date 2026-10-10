# -*- coding: utf-8 -*-
"""结构感知切片：把逐页解析结果切分为可检索的知识块。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

块类型：
    - text  ：正文段落切片（带重叠，保证跨句语义完整）
    - table ：表格键值对块（字段型问题的关键来源）

每个块都携带 source（源文件）/ page（页码）/ kind（类型）等元数据，
用于答案溯源与文档级路由。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src import config
from src.pdf_parser import PageData


@dataclass
class Chunk:
    """知识块。"""
    id: int
    text: str
    source: str
    page: int
    kind: str          # text / table

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "source": self.source,
                "page": self.page, "kind": self.kind}


# 章节标题：一、/（一）/ 1、/ 1.1 等，用于在切片时保留小节上下文
_SECTION_RE = re.compile(r"^(第[一二三四五六七八九十]+[节章]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+(\.\d+)*[、.])\s*\S")


def _split_long(text: str, size: int, overlap: int) -> List[str]:
    """按句末标点优先切分，超长再硬切，带重叠。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    # 先按段落
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    pieces: List[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 1 <= size:
            buf = (buf + "\n" + para) if buf else para
            continue
        if buf:
            pieces.append(buf)
        # 单段超长 → 按句子切
        if len(para) <= size:
            buf = para
            continue
        sents = re.split(r"(?<=[。！？；])", para)
        buf = ""
        for s in sents:
            if len(buf) + len(s) <= size:
                buf += s
            else:
                if buf:
                    pieces.append(buf)
                buf = s
        if buf:
            pieces.append(buf)
            buf = ""
    if buf:
        pieces.append(buf)
    # 加重叠
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for i in range(1, len(pieces)):
        tail = pieces[i - 1][-overlap:]
        out.append((tail + pieces[i]) if tail else pieces[i])
    return out


def build_chunks(pages: List[PageData]) -> List[Chunk]:
    """把逐页数据切分为知识块列表。"""
    chunks: List[Chunk] = []
    cid = 0
    section = ""
    for pg in pages:
        # 更新小节标题（跨页延续）
        for line in (pg.text or "").split("\n"):
            line = line.strip()
            if _SECTION_RE.match(line) and len(line) <= 40:
                section = line
        # 正文块
        for piece in _split_long(pg.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            if len(piece) < config.MIN_CHUNK_SIZE:
                continue
            body = f"【{section}】{piece}" if section else piece
            chunks.append(Chunk(id=cid, text=body, source=pg.source,
                                page=pg.page, kind="text"))
            cid += 1
        # 表格块
        for tbl in pg.tables:
            for piece in _split_long(tbl, config.CHUNK_SIZE, 0):
                if len(piece) < config.MIN_CHUNK_SIZE:
                    continue
                chunks.append(Chunk(id=cid, text=f"【表格】{piece}", source=pg.source,
                                    page=pg.page, kind="table"))
                cid += 1
    return chunks