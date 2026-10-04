# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：分块（Markdown 标题层级 + 表格/公式独立 + 父子块）

为什么不用 RecursiveCharacterTextSplitter 一刀切（工单明确要求）：
  招股说明书是强层级文档（章节 → 小节 → 段落）。固定窗口会把标题与正文、
  表头与表体切断，检索命中的片段常缺少"属于哪一节"的上下文。
  本模块做法：
    1. 按标题层级（text_level）切"节"（section），维护 heading_path；
    2. 节内正文按句子边界贪心打包成 300~450 字的**子块**（检索/向量化单位），
       同一节的子块共享**父块**（整节文本，≤1800 字，作为送 LLM 的上下文）；
    3. 表格/公式/图片**独立成块**，不参与正文拼接，block_type 标注；
    4. 元数据（page_idx / heading_path / source / block_type）全部写入 chunk，
       建库时进 Qdrant payload，供过滤与引用。

chunk 结构：
  {"chunk_id", "text", "parent_id", "parent_text", "parent_page_idx",
   "page_idx", "heading_path", "source", "block_type", "seq"}
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
from typing import Sequence

from src import config

_SENT_RE = re.compile(r"[^。！？；\n]*[。！？；\n]|[^。！？；\n]+$")

# 独立成块的类型（表格/公式/图片不并入正文流）
_STANDALONE_TYPES = {"table", "equation", "image"}


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句（保留标点），无标点长文本退化为整体。"""
    parts = [m.group(0) for m in _SENT_RE.finditer(text or "")]
    return [p for p in parts if p.strip()] or ([text] if text.strip() else [])


def _truncate_table(text: str, limit: int) -> str:
    """表格超长时保留首部（表头/表体前段）与尾部，中间用省略标记。"""
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    tail = text[-150:]
    return f"{head}\n…（表格过长，中间省略 {len(text) - len(head) - len(tail)} 字）\n{tail}"


def _pack_child_blocks(paragraphs: Sequence[tuple[str, int]], max_chars: int,
                       min_chars: int) -> list[tuple[str, int]]:
    """把 (段落文本, 页码) 序列打包成子块，返回 [(子块文本, 起始页码)]。

    页码归属说明：每个子块记录**自身起点所在页**（此前统一标记 section
    起始页，导致引用页码偏移与页覆盖统计失真，2026-10-04 测试发现并修复）。
    """
    blocks: list[tuple[str, int]] = []
    buf = ""
    buf_page = 0
    for para, page in paragraphs:
        for sent in _split_sentences(para):
            # 硬切分兜底：无句末标点的超长"句子"（如整段无标点的表格文本）
            pieces = ([sent[i:i + max_chars] for i in range(0, len(sent), max_chars)]
                      if len(sent) > max_chars else [sent])
            for piece in pieces:
                if buf and len(buf) + len(piece) > max_chars:
                    blocks.append((buf.strip(), buf_page))
                    buf = piece
                    buf_page = page
                else:
                    if not buf:
                        buf_page = page
                    buf += piece
        # 段落结束补换行，保持可读性
        if buf and not buf.endswith("\n"):
            buf += "\n"
    if buf.strip():
        blocks.append((buf.strip(), buf_page))

    # 合并过短块（< min_chars）到前一块，避免碎片；但不得超过 max_chars
    merged: list[tuple[str, int]] = []
    for b, page in blocks:
        if (merged and len(b) < min_chars
                and len(merged[-1][0]) + len(b) + 1 <= max_chars):
            merged[-1] = ((merged[-1][0] + "\n" + b).strip(), merged[-1][1])
        else:
            merged.append((b, page))
    return merged


def _iter_sections(items: Sequence[dict]) -> list[dict]:
    """按标题层级把条目切成节。

    返回 [{"heading_path": [...], "items": [...], "page_idx": int}, ...]
    标题条目本身也放入 items[0]（父块需要包含标题文本）。
    """
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []          # [(level, title)]
    cur: dict | None = None

    def _new_section() -> dict:
        return {"heading_path": [t for _l, t in stack],
                "items": [], "page_idx": 0}

    for it in items:
        is_heading = (it["type"] == "text" and it.get("text_level")
                      and 1 <= int(it["text_level"]) <= 6)
        if is_heading:
            lvl = int(it["text_level"])
            # 关闭当前节
            if cur is not None:
                sections.append(cur)
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, it["text"].strip()))
            cur = _new_section()
            cur["page_idx"] = it["page_idx"]
            cur["items"].append(it)
        else:
            if cur is None:                    # 文档开头无标题的正文
                cur = _new_section()
                cur["page_idx"] = it["page_idx"]
            cur["items"].append(it)
    if cur is not None:
        sections.append(cur)

    # 合并过短节（< SECTION_MIN_CHARS 且无独立价值）到下一节，减少碎片
    merged: list[dict] = []
    for sec in sections:
        body_len = sum(len(i["text"]) for i in sec["items"])
        if merged and body_len < config.SECTION_MIN_CHARS:
            prev = merged[-1]
            # 若本节无标题（heading_path 与上节相同），直接并入
            if sec["heading_path"] == prev["heading_path"]:
                prev["items"].extend(sec["items"])
                continue
        merged.append(sec)
    return merged


def _section_parent_text(sec: dict) -> str:
    """父块文本 = 节内所有条目文本（表格转 Markdown 原样保留），≤ PARENT_MAX_CHARS。"""
    parts: list[str] = []
    for it in sec["items"]:
        t = it["text"]
        if it["type"] == "table":
            t = _truncate_table(t, config.TABLE_MAX_CHARS)
        parts.append(t)
    text = "\n".join(p for p in parts if p).strip()
    if len(text) > config.PARENT_MAX_CHARS:
        text = text[: config.PARENT_MAX_CHARS] + "…"
    return text


def build_chunks(items: Sequence[dict], source: str) -> list[dict]:
    """条目列表 → 父子块 chunk 列表（chunk_id 稳定可复现）。"""
    chunks: list[dict] = []
    seq = 0
    for s_idx, sec in enumerate(_iter_sections(items), start=1):
        parent_id = f"s{s_idx:06d}"
        parent_text = _section_parent_text(sec)
        heading_path = sec["heading_path"]

        # 1) 独立块：表格 / 公式 / 图片
        body_paras: list[tuple[str, int]] = []
        for it in sec["items"]:
            if it["type"] in _STANDALONE_TYPES:
                text = it["text"]
                if it["type"] == "table":
                    text = _truncate_table(text, config.TABLE_MAX_CHARS)
                chunks.append({
                    "chunk_id": f"c{len(chunks) + 1:06d}",
                    "text": text,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "parent_page_idx": sec["page_idx"],
                    "page_idx": it["page_idx"],
                    "heading_path": list(heading_path),
                    "source": source,
                    "block_type": it["type"],
                    "seq": seq,
                })
                seq += 1
            else:
                body_paras.append((it["text"], int(it.get("page_idx", 0))))

        # 2) 正文子块（页码 = 子块自身起点所在页）
        for child, child_page in _pack_child_blocks(body_paras, config.CHILD_MAX_CHARS,
                                                    config.CHILD_MIN_CHARS):
            if not child.strip():
                continue
            chunks.append({
                "chunk_id": f"c{len(chunks) + 1:06d}",
                "text": child.strip(),
                "parent_id": parent_id,
                "parent_text": parent_text,
                "parent_page_idx": child_page,
                "page_idx": child_page,
                "heading_path": list(heading_path),
                "source": source,
                "block_type": "text",
                "seq": seq,
            })
            seq += 1
    return chunks


def stats(chunks: Sequence[dict]) -> dict:
    """分块统计（供建库日志与文档使用）。"""
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["block_type"]] = by_type.get(c["block_type"], 0) + 1
    parents = {c["parent_id"] for c in chunks}
    lens = [len(c["text"]) for c in chunks] or [0]
    return {
        "n_chunks": len(chunks),
        "n_parents": len(parents),
        "by_type": by_type,
        "avg_chars": round(sum(lens) / len(lens), 1),
        "max_chars": max(lens),
    }
