"""文档分块：固定长度 / 句子 / 段落 / 标题 / 语义，并支持父子块。"""
from __future__ import annotations

import re

from app.config import settings
from app.logging_conf import log

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")
_HEADING = re.compile(
    r"^\s*(第[一二三四五六七八九十百零〇\d]+[章节条款]|附\s*则|"
    r"[一二三四五六七八九十]+[、.．]|（[一二三四五六七八九十\d]+）|\(\d+\)|\d+[、.．])"
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _pack(units: list[str], size: int, overlap: int = 0) -> list[str]:
    """把句子/段落单元合并为不超过 size 的块，支持按字符重叠。"""
    chunks: list[str] = []
    buf: list[str] = []
    cur = 0
    for u in units:
        if cur and cur + len(u) > size:
            chunks.append("".join(buf))
            if overlap > 0:
                tail: list[str] = []
                tlen = 0
                for b in reversed(buf):
                    if tlen + len(b) > overlap:
                        break
                    tail.insert(0, b)
                    tlen += len(b)
                buf, cur = tail, tlen
            else:
                buf, cur = [], 0
        buf.append(u)
        cur += len(u)
    if buf:
        chunks.append("".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    out: list[str] = []
    start, n = 0, len(text)
    while start < n:
        out.append(text[start:start + size])
        start += max(size - overlap, 1)
    return [c.strip() for c in out if c.strip()]


def chunk_sentence(text: str, size: int, overlap: int) -> list[str]:
    return _pack(split_sentences(text), size=size, overlap=overlap)


def chunk_paragraph(text: str, size: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paras:
        if len(p) <= size:
            units.append(p)
        else:
            units.extend(split_sentences(p) or [p])
    return _pack(units, size=size)


def chunk_heading(text: str, size: int) -> list[str]:
    sections: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if _HEADING.match(line) and buf:
            sections.append("\n".join(buf))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append("\n".join(buf))

    out: list[str] = []
    for s in sections:
        s = s.strip()
        if not s:
            continue
        if len(s) <= size:
            out.append(s)
        else:
            packed = _pack(split_sentences(s), size=size)
            out.extend(packed or [s])
    return out


def chunk_semantic(text: str, size: int) -> list[str]:
    """基于相邻句向量相似度的语义分块；模型不可用时回退句子分块。"""
    sents = split_sentences(text)
    if len(sents) <= 1:
        return [text.strip()] if text.strip() else []
    try:
        import numpy as np

        from app.core.registry import get_embedder

        embedder = get_embedder()
        if not embedder.available:
            raise RuntimeError("embedder 不可用")
        vecs = np.asarray(embedder.encode(sents), dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        chunks: list[str] = []
        buf: list[str] = [sents[0]]
        cur = len(sents[0])
        for i in range(1, len(sents)):
            sim = float(np.dot(vecs[i - 1], vecs[i]))
            if (sim < 0.55 or cur + len(sents[i]) > size) and buf:
                chunks.append("".join(buf))
                buf, cur = [], 0
            buf.append(sents[i])
            cur += len(sents[i])
        if buf:
            chunks.append("".join(buf))
        return [c.strip() for c in chunks if c.strip()]
    except Exception as exc:  # noqa: BLE001
        log.warning("语义分块不可用，回退句子分块: %s", exc)
        return chunk_sentence(text, size, 0)


def chunk_text(text: str, strategy: str | None = None, size: int | None = None,
               overlap: int | None = None) -> list[dict]:
    strategy = strategy or settings.chunk_strategy
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap

    if strategy == "fixed":
        parts = chunk_fixed(text, size, overlap)
    elif strategy == "paragraph":
        parts = chunk_paragraph(text, size)
    elif strategy == "heading":
        parts = chunk_heading(text, size)
    elif strategy == "semantic":
        parts = chunk_semantic(text, size)
    else:  # sentence
        parts = chunk_sentence(text, size, overlap)

    return [{"text": p.strip(), "chunk_type": strategy} for p in parts if p.strip()]


def chunk_parent_child(text: str, strategy: str | None = None, size: int | None = None,
                       overlap: int | None = None, parent_size: int | None = None) -> list[dict]:
    """父子块：父块按段落聚合，子块继续细分，子块携带 parent 原文。"""
    strategy = strategy or settings.chunk_strategy
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    parent_size = parent_size or size * 3

    parents = chunk_text(text, strategy="paragraph", size=parent_size)
    children: list[dict] = []
    for parent in parents:
        for child in chunk_text(parent["text"], strategy=strategy, size=size, overlap=overlap):
            children.append(
                {"text": child["text"], "chunk_type": child["chunk_type"], "parent": parent["text"]}
            )
    return children or chunk_text(text, strategy, size, overlap)
