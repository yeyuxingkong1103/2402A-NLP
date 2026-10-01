# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：分块（Chunking）

需求（工单第三步）：
  - 按 Markdown 标题层级切分
  - 表格独立成块
  - 每个 chunk 携带 page_idx、heading_path、source

设计要点：
  1. **不跨标题切分**：同一个 heading_path 下的内容才允许合并进同一块，
     保证每个块语义内聚，引用时能给出准确的章节路径。
  2. **表格独立成块**：招股书的核心事实（注册资本、募集资金、收入构成）
     大多在表格里，表格被切碎会直接导致答不出。表格块保留完整 Markdown，
     并补上表头行，避免跨行断表。
  3. **标题上下文注入**：把 heading_path 作为前缀写进 embedding 文本。
     例如「（一）公司基本情况 > 注册资本」这样的前缀能让
     "注册资本是多少" 这类问句与块之间的语义距离显著缩小。
  4. **句级打包 + 重叠**：按中文句末标点切句后贪心装箱，
     相邻块保留 CHUNK_OVERLAP_CHARS 重叠，避免答案正好被切在边界上。
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401

import argparse
import json
import os
import re
from typing import Iterable

from src import config

# 中文句末标点（用于切句）
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*")
# 连续空白压缩
WS_RE = re.compile(r"[ \t　]+")


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """压缩空白，保留换行语义。"""
    text = text.replace("\r", "")
    text = WS_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """按中文句末标点切句；无标点的长串会保持为一句（后续再强制切）。"""
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def force_split(sentence: str, max_chars: int) -> list[str]:
    """对超长且无句末标点的句子做强制切分（优先在逗号/顿号处断）。"""
    if len(sentence) <= max_chars:
        return [sentence]
    out, buf = [], ""
    # 次级断点：逗号、顿号、冒号、换行
    pieces = re.split(r"(?<=[，、：,:\n])", sentence)
    for p in pieces:
        if len(buf) + len(p) <= max_chars:
            buf += p
        else:
            if buf:
                out.append(buf)
            # 单段仍超长 -> 硬切
            while len(p) > max_chars:
                out.append(p[:max_chars])
                p = p[max_chars:]
            buf = p
    if buf:
        out.append(buf)
    return out


def pack_sentences(sentences: Iterable[str], max_chars: int,
                   overlap_chars: int) -> list[str]:
    """把句子贪心装箱成不超过 max_chars 的块，块间带 overlap。"""
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if not sent:
            continue
        if len(buf) + len(sent) <= max_chars:
            buf += sent
        else:
            if buf.strip():
                chunks.append(buf.strip())
                # 生成重叠尾巴：取上一块结尾若干字符
                tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
                buf = tail + sent
            else:
                buf = sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def make_context_prefix(heading_path: list[str], max_part: int = 24) -> str:
    """构造注入到 embedding 文本前的章节上下文。

    只保留最后 3 级，并把每一级截断到 max_part 字符 —— 否则个别超长
    "标题"（其实是误判的正文行）会让前缀喧宾夺主，稀释正文的向量语义。
    """
    if not heading_path:
        return ""
    tail = [h.strip() for h in heading_path if h and h.strip()][-3:]
    tail = [h if len(h) <= max_part else h[:max_part] + "…" for h in tail]
    return " > ".join(tail)


# 目录行特征：大量连续点号 + 结尾页码，例如 "十二、…独立董事意见 .... 269"
TOC_LINE_RE = re.compile(r"[.．·]{4,}\s*\d+\s*$")
# 纯页码行，例如 "1-1-60"
PAGE_NO_RE = re.compile(r"^\s*\d+[-－]\d+[-－]\d+\s*$")


def is_toc_line(text: str) -> bool:
    """判断是否为目录行（对问答无价值，且会污染检索）。"""
    t = text.strip()
    return bool(TOC_LINE_RE.search(t) or PAGE_NO_RE.match(t))


# ---- 页眉/页脚/续表标记等"版式噪声"识别 -------------------------------------
# 招股书每页页眉都是「武汉兴图新科电子股份有限公司 招股意向书」，
# 每张续表前都有「续表：」。这些行本身没有信息量，但它们**含有公司全称**，
# 而公司全称又几乎出现在每个用户问题里 —— 结果就是 BM25 把几十个页眉块
# 全部排到最前，真正含答案的段落反而被挤出 top-k。
# 识别方式不靠硬编码字符串，而是数据驱动：在同一文档里短文本反复出现
# （≥ BOILERPLATE_MIN_REPEATS 次，且分布在 ≥ BOILERPLATE_MIN_PAGES 个页面上），
# 就判定为版式噪声。
BOILERPLATE_MIN_REPEATS = 8
BOILERPLATE_MIN_PAGES = 8
BOILERPLATE_MAX_CHARS = 60


def find_boilerplate(content: list[dict],
                     min_repeats: int = BOILERPLATE_MIN_REPEATS,
                     min_pages: int = BOILERPLATE_MIN_PAGES,
                     max_chars: int = BOILERPLATE_MAX_CHARS) -> set[str]:
    """找出全文反复出现的短文本（页眉、页脚、续表标记等）。"""
    counts: dict[str, int] = {}
    pages: dict[str, set] = {}
    for item in content:
        if item.get("type") != "text":
            continue
        text = normalize(item.get("text", ""))
        if not text or len(text) > max_chars:
            continue
        counts[text] = counts.get(text, 0) + 1
        pages.setdefault(text, set()).add(item.get("page_idx"))

    return {t for t, n in counts.items()
            if n >= min_repeats and len(pages[t]) >= min_pages}


# ---------------------------------------------------------------------------
# 主分块逻辑
# ---------------------------------------------------------------------------
def chunk_content_list(content: list[dict], max_chars: int | None = None,
                       overlap_chars: int | None = None,
                       source: str | None = None) -> list[dict]:
    """把 content_list 切成带元数据的 chunk 列表。"""
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    overlap_chars = overlap_chars if overlap_chars is not None else config.CHUNK_OVERLAP_CHARS
    source = source or config.SOURCE_NAME

    chunks: list[dict] = []
    seq = 0
    boilerplate = find_boilerplate(content)

    # ---- 1) 按 heading_path 归组连续正文 ----
    groups: list[dict] = []
    cur: dict | None = None

    def flush() -> None:
        nonlocal cur
        if cur and cur["texts"]:
            groups.append(cur)
        cur = None

    for item in content:
        hp = tuple(item.get("heading_path") or [])
        if item["type"] == "table":
            flush()
            groups.append({
                "heading_path": list(hp),
                "page_idx": item["page_idx"],
                "texts": [],
                "table": item.get("table_body", ""),
            })
            continue

        text = normalize(item.get("text", ""))
        if not text or is_toc_line(text) or text in boilerplate:
            continue

        # 标题行归入"它自己开启的那一节"，与随后的正文同一个分组。
        # 标题在 parse 阶段写出的 heading_path 不含自己，
        # 因此这里把标题文本本身补进路径，才能与后续正文对齐。
        key = hp + (text,) if item.get("is_heading") else hp

        if cur is None or cur["heading_path"] != list(key):
            flush()
            cur = {"heading_path": list(key), "page_idx": item["page_idx"],
                   "texts": [], "table": None}
        cur["texts"].append(text)

    flush()

    # ---- 1.5) 合并"只有标题"的孤立分组 ----
    # 标题行会自成一组；若它后面紧跟表格（表格强制独立成组），
    # 就会留下一个几字长的纯标题块（如「本次发行概况」「目 录」）。
    # 这类块检索价值极低却会占据召回名额，这里把它们并入下一组。
    merged_groups: list[dict] = []
    pending_head: list[str] = []
    for g in groups:
        is_heading_only = (not g["table"] and len(g["texts"]) == 1
                           and g["heading_path"]
                           and g["texts"][0] == g["heading_path"][-1])
        if is_heading_only:
            pending_head.append(g["texts"][0])
            continue
        if pending_head:
            g["texts"] = pending_head + g["texts"]
            pending_head = []
        merged_groups.append(g)
    # 末尾若还挂着纯标题（文档结束），保留为独立块，避免丢信息
    if pending_head:
        merged_groups.append({"heading_path": pending_head, "page_idx": 0,
                              "texts": pending_head, "table": None})
    groups = merged_groups

    # ---- 2) 逐组切块 ----
    for g in groups:
        prefix = make_context_prefix(g["heading_path"])
        head = f"【{prefix}】\n" if prefix else ""

        if g["table"]:  # --- 表格独立成块 ---
            body = g["table"].strip()
            # 超长表格：保留前 TABLE_MAX_CHARS，并显式说明被截断
            truncated = False
            if len(body) > config.TABLE_MAX_CHARS:
                cut = body[:config.TABLE_MAX_CHARS]
                body = cut.rsplit("\n", 1)[0] + "\n（表格较长，此处截断）"
                truncated = True
            seq += 1
            chunks.append({
                "chunk_id": f"c{seq:05d}",
                "type": "table",
                "text": head + body,
                "raw_text": body,
                "page_idx": g["page_idx"],
                "heading_path": g["heading_path"],
                "source": source,
                "truncated": truncated,
            })
            continue

        # --- 正文：句级打包 ---
        joined = "\n".join(g["texts"])
        sentences: list[str] = []
        for s in split_sentences(joined):
            sentences.extend(force_split(s, max_chars))

        for piece in pack_sentences(sentences, max_chars, overlap_chars):
            if len(piece) < config.CHUNK_MIN_CHARS and chunks:
                # 碎块并入上一块（仅当同章节同类型）
                prev = chunks[-1]
                if prev["type"] == "text" and prev["heading_path"] == g["heading_path"]:
                    prev["text"] += piece
                    prev["raw_text"] += piece
                    continue
            seq += 1
            chunks.append({
                "chunk_id": f"c{seq:05d}",
                "type": "text",
                "text": head + piece,
                "raw_text": piece,
                "page_idx": g["page_idx"],
                "heading_path": g["heading_path"],
                "source": source,
            })

    return chunks


# ---------------------------------------------------------------------------
# 统计与 CLI
# ---------------------------------------------------------------------------
def summarize(chunks: list[dict]) -> dict:
    """汇总分块统计信息。"""
    n = len(chunks)
    n_tab = sum(1 for c in chunks if c["type"] == "table")
    lens = [len(c["raw_text"]) for c in chunks] or [0]
    return {
        "total": n,
        "text": n - n_tab,
        "table": n_tab,
        "avg_chars": round(sum(lens) / max(len(lens), 1), 1),
        "max_chars": max(lens),
        "min_chars": min(lens),
        "truncated_tables": sum(1 for c in chunks if c.get("truncated")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="content_list.json -> chunks.json")
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(config.PARSED_DIR,
                                         f"{os.path.splitext(config.SOURCE_NAME)[0]}_content_list.json"))
    ap.add_argument("--out", default=os.path.join(config.CHUNK_DIR, "chunks.json"))
    ap.add_argument("--max-chars", type=int, default=config.CHUNK_MAX_CHARS)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP_CHARS)
    args = ap.parse_args()

    print(f"读取: {args.inp}")
    with open(args.inp, encoding="utf-8") as fh:
        content = json.load(fh)

    boiler = find_boilerplate(content)
    if boiler:
        print(f"  识别到版式噪声（页眉/页脚/续表标记）{len(boiler)} 类，已剔除")

    chunks = chunk_content_list(content, args.max_chars, args.overlap)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=1)

    st = summarize(chunks)
    print(f"\n分块完成 -> {args.out}")
    print(f"  总块数      : {st['total']}（正文 {st['text']} / 表格 {st['table']}）")
    print(f"  平均长度    : {st['avg_chars']} 字符")
    print(f"  最长/最短   : {st['max_chars']} / {st['min_chars']}")
    if st["truncated_tables"]:
        print(f"  被截断表格  : {st['truncated_tables']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
