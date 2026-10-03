# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：chunk —— 标题感知滑窗分块
# 说明：先按章节标题（第X节 / 一、 / （一） / MinerU text_level）聚合为“小节”，
#       再在小节内做 600/120 滑窗。每块带 page / section / type 元数据，供引用溯源。

import re

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK

_SEC_RE = re.compile(r"^第[一二三四五六七八九十百]+节")
_H1_RE = re.compile(r"^[一二三四五六七八九十]+、")
_H2_RE = re.compile(r"^（[一二三四五六七八九十]+）")
_NUM_RE = re.compile(r"^\d+(\.\d+)*[、．.]?")


def is_heading(b):
    """判定 block 是否为标题（用于聚合小节）。"""
    t = b.get("text", "")
    if b.get("type") != "text":
        return False
    if b.get("level"):
        return True
    if len(t) > 40:
        return False
    return bool(_SEC_RE.match(t) or _H1_RE.match(t) or _H2_RE.match(t))


def _sliding(text, size, overlap):
    if len(text) <= size:
        return [text]
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += step
    return out


def build_chunks(blocks):
    chunks = []
    section = ""
    buf = []          # list[(text, page)]
    buf_type = "text"
    cid = 0

    def flush():
        nonlocal buf, buf_type, cid
        if not buf:
            return
        # 拼接文本并记录每个 block 的字符区间 -> 页码，实现“每块页码精准归属”
        parts, spans = [], []
        pos = 0
        for txt, pg in buf:
            if pos:
                parts.append("\n")
                pos += 1
            spans.append((pos, pos + len(txt), pg))
            parts.append(txt)
            pos += len(txt)
        full = "".join(parts)
        if len(full) < MIN_CHUNK:
            buf, buf_type = [], "text"
            return

        def pages_for(a, b):
            ps = [pg for (s, e, pg) in spans if e > a and s < b and pg]
            return (min(ps), max(ps)) if ps else (0, 0)

        if len(full) <= CHUNK_SIZE:
            pieces = [(0, len(full))]
        else:
            step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            pieces = [(i, min(i + CHUNK_SIZE, len(full))) for i in range(0, len(full), step)]
        for a, b in pieces:
            piece = full[a:b].strip()
            if not piece or (len(piece) < MIN_CHUNK and chunks):
                continue
            p0, p1 = pages_for(a, b)
            chunks.append({"id": cid, "text": piece, "page": p0, "page_end": p1,
                           "section": section, "type": buf_type})
            cid += 1
        buf, buf_type = [], "text"

    for b in blocks:
        if is_heading(b):
            flush()
            section = b["text"].strip()[:60]
            continue
        if b.get("type") == "table":
            flush()
            chunks.append({"id": cid, "text": b["text"], "page": b["page"],
                           "page_end": b["page"], "section": section, "type": "table"})
            cid += 1
            continue
        buf.append((b["text"], b.get("page", 0)))
    flush()
    return chunks


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from parse import load_blocks
    from clean import clean_blocks
    ch = build_chunks(clean_blocks(load_blocks()))
    print("chunks:", len(ch))
    for c in ch[:5]:
        print("p%d-%d [%s] %s" % (c["page"], c["page_end"], c["section"], c["text"][:80]))
