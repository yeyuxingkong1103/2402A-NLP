"""产出：按 docs/04 §7 契约构造 chunk、渲染自检报告、原子落盘。"""

from __future__ import annotations

import json
import os

from .core import (
    EXIT_BAD_INPUT, HIGH, LOW, RULE_VERSION, ChunkError, char_len,
)


def build_chunk(doc_id: str, file_name: str, source_hash: str,
                seq: int, blocks: list[dict]) -> dict:
    text = "\n\n".join(b["text"] for b in blocks)
    head = blocks[0]["heading_path"]
    pages = [b["page"] for b in blocks]
    return {
        "chunk_id": "%s:%04d" % (doc_id, seq),
        "doc_id": doc_id,
        "file_name": file_name,
        # text 是纯正文原文，一字不改；text_for_embedding 只是给检索加了章节
        # 路标（heading_path 前缀），**仅用于生成向量，不展示、不入库**。
        # 用户核对引用时面对的是 text，与 PDF 原文逐字一致——两者不可混为一谈。
        "text": text,
        "text_for_embedding": ("%s\n\n%s" % (" > ".join(head), text)) if head else text,
        "page_start": min(pages),
        "page_end": max(pages),
        "section": head[-1] if head else "",
        "heading_path": head,
        "block_type": blocks[0]["block_type"],
        "block_ids": list(dict.fromkeys(b["block_id"] for b in blocks)),
        "sub_index": blocks[0].get("_sub", 0),
        "char_len": char_len(text),
        "source_hash": source_hash,
        "chunk_rule_version": RULE_VERSION,
    }


def render_report(records: list[dict], decisions, crossed: int, source_hash: str) -> str:
    sizes = sorted(r["char_len"] for r in records)
    n = len(sizes) or 1
    below = [r for r in records if r["char_len"] < LOW]
    over = [r for r in records if r["char_len"] > HIGH]
    nopage = [r for r in records if not r.get("page_start") or not r.get("page_end")]
    answered = len(decisions.items) - len(decisions.pending())

    lines = ["", "=" * 68, "分块自检指标（docs/04 §7）", "=" * 68, ""]
    lines.append("  chunk 总数               %d" % len(records))
    lines.append("  长度中位数               %d 字   （目标 %d-%d）"
                 % (sizes[len(sizes) // 2], LOW, HIGH))
    lines.append("  低于下限占比             %.1f%%   （%d 个；docs/04 门槛 <15%%）"
                 % (100.0 * len(below) / n, len(below)))
    lines.append("  高于上限占比             %.1f%%   （%d 个）"
                 % (100.0 * len(over) / n, len(over)))
    lines.append("  页码缺失的 chunk 数      %d       （必须为 0）" % len(nopage))
    lines.append("  跨父章节合并次数         %d       （D1 只允许同父章节）" % crossed)
    lines.append("  待裁决项                 %d       （已答复 %d）"
                 % (len(decisions.items), answered))
    lines.append("  source_hash              %s" % source_hash[:16])
    lines.append("")
    lines.append("  长度分布：")
    for lo, hi in [(0, 200), (200, 500), (500, 700), (700, 1500), (1500, 10 ** 9)]:
        lines.append("    [%5d,%5d)  %3d 个"
                     % (lo, hi, sum(1 for r in records if lo <= r["char_len"] < hi)))
    if over:
        lines.append("")
        lines.append("  超长 chunk（表格整表成块属预期；正文超长应已按句边界切开）：")
        for r in sorted(over, key=lambda x: -x["char_len"])[:8]:
            lines.append("    %-18s %5d 字  [%-6s] %s"
                         % (r["chunk_id"], r["char_len"], r["block_type"],
                            r["text"][:40].replace("\n", " ")))
    lines.append("")
    return "\n".join(lines)


def write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_text(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    os.replace(tmp, path)


def load_answers(out_dir: str, doc_id: str) -> dict:
    """人工答复：{decision_id: 选项}。不存在即视为全部未答复。"""
    path = os.path.join(out_dir, "%s.answers.json" % doc_id)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkError(EXIT_BAD_INPUT, "answers.json 读取失败：%s\n  %s" % (path, exc)) from exc
