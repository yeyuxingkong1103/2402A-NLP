# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：上下文组装与压缩（小块检索 → 大块上下文）

工单02 要求"多路召回 + 融合，最终给 LLM 的 context 要控制长度"：
  检索命中的是 300~450 字的**子块**（精确），但只给子块会丢上下文
  （数字脱离"报告期内/2018年度"的限定就失去意义）。
  本模块把命中子块映射回其**父块**（所属小节全文，≤1800 字）：
    1. 按 parent_id 去重（同一节命中多个子块只保留一份父块）；
    2. 父块内以命中子块为中心裁剪到 per_parent_chars（保留邻域句子）；
    3. 全部片段按最佳名次排序，总长受 max_chars 限制；
    4. 输出带【片段N】编号 + 页码 + 标题路径，供 LLM 生成 [n] 引用。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

from typing import Sequence

from src import config


def _clip_around(text: str, anchor: str, limit: int) -> str:
    """在 text 中以 anchor 为中心裁剪到 limit 字符（句子边界优先）。"""
    if len(text) <= limit:
        return text
    pos = text.find(anchor[:40]) if anchor else -1
    if pos < 0:
        return text[:limit] + "…"
    half = max((limit - len(anchor)) // 2, 100)
    start = max(pos - half, 0)
    end = min(pos + len(anchor) + half, len(text))
    # 向前扩到句首
    for sep in ("。", "；", "\n"):
        p = text.rfind(sep, max(start - 80, 0), start)
        if p >= 0:
            start = p + 1
            break
    out = text[start:end]
    return ("…" if start > 0 else "") + out + ("…" if end < len(text) else "")


def expand_parents(hits: Sequence[dict], max_chars: int | None = None,
                   per_parent_chars: int = 1200) -> list[dict]:
    """命中子块 → 去重后的父块片段列表（有序）。

    返回项：{"parent_id", "parent_text"(裁剪后), "page_idx", "heading_path",
             "source", "child_hits": [子块…], "best_rank"}
    """
    max_chars = max_chars or config.CONTEXT_MAX_CHARS
    order: list[str] = []
    groups: dict[str, dict] = {}

    for rank, h in enumerate(hits, start=1):
        pid = h.get("parent_id") or h.get("chunk_id") or f"r{rank}"
        g = groups.get(pid)
        if g is None:
            is_table = str(h.get("block_type", "")).startswith("table") or bool(
                h.get("table_id"))
            g = {
                "parent_id": pid,
                "parent_text": h.get("parent_text") or h.get("text", ""),
                "page_idx": int(h.get("parent_page_idx", h.get("page_idx", 0))),
                "heading_path": list(h.get("heading_path") or []),
                "source": h.get("source", config.SOURCE_NAME),
                "child_hits": [],
                "best_rank": rank,
                "best_child_text": h.get("text", ""),
                # --- 工单3：表格上下文（整表还原）标记与元数据 ---
                "is_table": is_table,
                "table_id": h.get("table_id", ""),
                "table_caption": h.get("table_caption", ""),
                "table_header": h.get("table_header", ""),
                "table_html": h.get("table_html", ""),
                "block_type": h.get("block_type", "text"),
            }
            groups[pid] = g
            order.append(pid)
        g["child_hits"].append(h)
        g["best_rank"] = min(g["best_rank"], rank)

    # 按最佳名次排序，逐段装入预算。
    # 同页去重只对正文生效：表格块**不参与**同页去重——同一页常有多张语义不同的表
    # （如"本次发行概况"与"募集资金运用"同页），去重会丢掉恰好命中的那张。
    out: list[dict] = []
    used = 0
    seen_pages: set[int] = set()
    for pid in sorted(order, key=lambda p: groups[p]["best_rank"]):
        g = groups[pid]
        page = int(g.get("page_idx", -1))
        if (config.CONTEXT_DEDUP_SAME_PAGE and not g.get("is_table")
                and page in seen_pages):
            continue
        remaining = max_chars - used
        if remaining < 200:                     # 预算不足，停止装入
            break
        if g.get("is_table"):
            # 表格题必须喂**整表结构**（表头 + 全部行）：工单明确要求
            # "LLM 生成时要把表格结构一起喂进去，而不是只喂文本片段"。
            # 故不做"以命中行居中裁剪"，只按表格预算截断。
            text = g["parent_text"][: min(config.TABLE_CTX_MAX_CHARS, remaining)]
        else:
            text = _clip_around(g["parent_text"], g["best_child_text"],
                                min(per_parent_chars, remaining))
        g["parent_text"] = text
        used += len(text)
        seen_pages.add(page)
        out.append(g)
    return out


def build_context(blocks: Sequence[dict]) -> str:
    """父块片段 → LLM 上下文文本（带【片段N】/页码/标题路径，供引用）。

    工单3：表格片段额外标注「表格」并给出表名，提示 LLM 这是结构化表格数据，
    需要按列名读取、必要时做行间计算（如"占发行后总股本的比例"= 发行股数 ÷ 发行后总股本）。
    """
    parts: list[str] = []
    for i, b in enumerate(blocks, start=1):
        heading = " / ".join(b.get("heading_path") or []) or "（无标题）"
        page = int(b.get("page_idx", 0)) + 1     # page_idx 0 基 → 展示 1 基
        tag = ""
        if b.get("is_table"):
            cap = b.get("table_caption") or "未命名表"
            tag = f"【表格数据｜{cap}】"
        parts.append(
            f"【片段{i}】来源：{b.get('source', config.SOURCE_NAME)} "
            f"第{page}页（{heading}）{tag}\n{b.get('parent_text', '').strip()}"
        )
    return "\n\n---\n\n".join(parts)


def citation_map(blocks: Sequence[dict]) -> list[dict]:
    """片段编号 → 页码/标题/表格定位，供答案引用与界面展示（工单3 增加表格字段）。"""
    out = []
    for i, b in enumerate(blocks, start=1):
        out.append({
            "index": i,
            "page_idx": int(b.get("page_idx", 0)),
            "page_display": int(b.get("page_idx", 0)) + 1,
            "heading_path": list(b.get("heading_path") or []),
            "source": b.get("source", config.SOURCE_NAME),
            "parent_id": b.get("parent_id"),
            # --- 工单3：表格定位（哪个 PDF / 哪一页 / 哪个表 / 哪一行）---
            "is_table": bool(b.get("is_table")),
            "table_id": b.get("table_id", ""),
            "table_caption": b.get("table_caption", ""),
            "table_header": b.get("table_header", ""),
            "row_indices": sorted({int(h.get("row_index", -1))
                                   for h in b.get("child_hits", [])
                                   if int(h.get("row_index", -1)) >= 0}),
        })
    return out
