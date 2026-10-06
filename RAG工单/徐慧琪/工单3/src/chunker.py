# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：分块（Markdown 标题层级 + 表格结构化 + 父子块）

为什么不用 RecursiveCharacterTextSplitter 一刀切（工单明确要求）：
  招股说明书是强层级文档（章节 → 小节 → 段落）。固定窗口会把标题与正文、
  表头与表体切断，检索命中的片段常缺少"属于哪一节"的上下文。
  本模块做法：
    1. 按标题层级（text_level）切"节"（section），维护 heading_path；
    2. 节内正文按句子边界贪心打包成 300~450 字的**子块**（检索/向量化单位），
       同一节的子块共享**父块**（整节文本，≤1800 字，作为送 LLM 的上下文）；
    3. 公式/图片独立成块，不参与正文拼接，block_type 标注；
    4. 元数据（page_idx / heading_path / source / block_type）全部写入 chunk，
       建库时进 Qdrant payload，供过滤与引用。

工单3 变化（表格）：
  表格不再走"整表一个块"的旧路径，而是转交 `table_chunker`：
    每个表格 → 1 个表头块 + N 个行级块，parent_id = table_id，parent_text = 整表。
  于是表格与正文共享同一套"子块检索 + 父块上下文"机制，`context.expand_parents`
  无需特判即可把"命中某一行"还原为"整张表"送 LLM。
  表格也不再并入所属节的父块文本——否则一张长表会挤占 1800 字的正文上下文预算。

chunk 结构：
  {"chunk_id", "text", "parent_id", "parent_text", "parent_page_idx",
   "page_idx", "heading_path", "source", "block_type", "seq"}
  表格 chunk 另含：table_id / table_html / table_header / row_index /
                   table_caption / table_quality / table_n_rows / from_ocr
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re
from typing import Sequence

from src import config, table_chunker

_SENT_RE = re.compile(r"[^。！？；\n]*[。！？；\n]|[^。！？；\n]+$")

# 独立成块的类型（公式/图片不并入正文流；表格走 table_chunker 专项链路）
_STANDALONE_TYPES = {"equation", "image"}


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句（保留标点），无标点长文本退化为整体。"""
    parts = [m.group(0) for m in _SENT_RE.finditer(text or "")]
    return [p for p in parts if p.strip()] or ([text] if text.strip() else [])


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


def item_headings(items: Sequence[dict]) -> dict[int, list[str]]:
    """每个条目下标 → 其所属章节标题路径。

    给"无标题表格"补标题用（`table_chunker.resolve_caption`）：MinerU 的
    table_caption 常为空，用所属章节标题当表名比"（无标题）"信息量大得多。
    与 `_iter_sections` 用同一套标题栈规则，保证口径一致。
    """
    out: dict[int, list[str]] = {}
    stack: list[tuple[int, str]] = []
    for idx, it in enumerate(items):
        is_heading = (it["type"] == "text" and it.get("text_level")
                      and 1 <= int(it["text_level"]) <= 6)
        if is_heading:
            lvl = int(it["text_level"])
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, it["text"].strip()))
            out[idx] = [t for _l, t in stack[:-1]]
        else:
            out[idx] = [t for _l, t in stack]
    return out


def _section_parent_text(sec: dict) -> str:
    """父块文本 = 节内正文/公式/图片文本，≤ PARENT_MAX_CHARS。

    工单3：表格**不再并入**节父块。表格已由 table_chunker 生成独立的
    parent（parent_text = 整表），若再塞进节父块，一张长表会挤占 1800 字的
    正文上下文预算，导致正文被截断。
    """
    parts: list[str] = []
    for it in sec["items"]:
        if it["type"] == "table":
            continue
        parts.append(it["text"])
    text = "\n".join(p for p in parts if p).strip()
    if len(text) > config.PARENT_MAX_CHARS:
        text = text[: config.PARENT_MAX_CHARS] + "…"
    return text


def build_chunks(items: Sequence[dict], source: str,
                 ocr_pages: dict | None = None) -> list[dict]:
    """条目列表 → 父子块 chunk 列表（chunk_id 稳定可复现）。

    ocr_pages：{page_idx: OCR文本}，低质量表格页的 PaddleOCR-VL 兜底结果（可为空）。
    """
    chunks: list[dict] = []
    seq = 0
    ocr_pages = ocr_pages or {}
    # 每个 (source, page) 上的表格序号 → 构成文档内唯一的 table_id
    table_seq: dict[tuple[str, int], int] = {}

    for s_idx, sec in enumerate(_iter_sections(items), start=1):
        # 父块 ID 必须带 source：两文档各自从 s000001 开始，只按序号会在多文档
        # 场景下把"招股1 的某节"和"招股2 的某节"当成同一个父块合并（工单3 修正）。
        parent_id = f"{source}#s{s_idx:06d}"
        parent_text = _section_parent_text(sec)
        heading_path = sec["heading_path"]

        # 1) 独立块：表格（专项链路）/ 公式 / 图片
        body_paras: list[tuple[str, int]] = []
        for it in sec["items"]:
            if it["type"] == "table":
                page_idx = int(it.get("page_idx", 0))
                key = (source, page_idx)
                t_no = table_seq.get(key, 0)
                table_seq[key] = t_no + 1
                table_id = table_chunker.make_table_id(source, page_idx, t_no)
                chunks.extend(table_chunker.chunk_table_item(
                    it, source=source, heading_path=heading_path,
                    table_id=table_id, chunk_id_start=len(chunks) + 1,
                    seq_start=seq, ocr_text=ocr_pages.get(page_idx, "")))
                seq = len(chunks)
            elif it["type"] in _STANDALONE_TYPES:
                chunks.append({
                    "chunk_id": f"c{len(chunks) + 1:06d}",
                    "text": it["text"],
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
    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.get("source", "?")] = by_source.get(c.get("source", "?"), 0) + 1
    return {
        "n_chunks": len(chunks),
        "n_parents": len(parents),
        "by_type": by_type,
        "by_source": by_source,
        "avg_chars": round(sum(lens) / len(lens), 1),
        "max_chars": max(lens),
        "tables": table_chunker.stats(chunks),
    }
