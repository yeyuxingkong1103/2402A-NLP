# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
src/chunker_optimized.py —— 工单二语义分块器（父子块 + 语义边界 + 滑动窗口）

在工单一 src/chunker.py（RecursiveCharacterTextSplitter 逐页固定分块）基础上增强：
  1. 语义分块：基于 Step 2 版面分析的标题层级（layout_blocks.headings）切分章节，
     正文按段落边界聚合为语义完整父块，杜绝"句子被拦腰切断"
  2. 父子块（small-to-big）：父块 = 语义完整单元（章节段落/表格），用于 LLM 生成上下文；
     子块 = 滑动窗口碎片，用于向量检索匹配（检索命中子块 → 召回父块，兼顾精度与语境）
  3. 滑动窗口：chunk_size=600, overlap=100，切点对齐句末标点
  4. 元数据保留：page / heading / parent_id / chunk_type（另含 role 区分父子）

用法（项目根目录，无参数即执行工单验证命令）：
  python -m src.chunker_optimized
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# ---------- 工单参数（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
CHUNK_SIZE = 600        # 工单要求：滑动窗口窗口大小
CHUNK_OVERLAP = 100     # 工单要求：滑动窗口重叠
PARENT_MAX_CHARS = 1200  # 父块最大长度（超过则按段落边界再分）
SENT_END = "。！？；\n"   # 句末标点集合（切点对齐用）

DEFAULT_PARSED = "data/optimized/招股说明书1_optimized.json"
DEFAULT_BASELINE_CHUNKS = "data/chunks/招股说明书1_chunks.json"  # 工单一分块产物（对比用）
DEFAULT_OUT = "data/optimized/招股说明书1_chunks_optimized.json"


def _find_cut(text: str, target: int, window: int) -> int:
    """在 target 附近找句末标点作为切点（向前最多找 window//2），保证子块句子完整"""
    lo = max(1, target - window // 2)
    for i in range(min(target, len(text) - 1), lo, -1):
        if text[i - 1] in SENT_END:
            return i
    return target  # 无标点（纯数字/表格行）时按窗口硬切


def sliding_window(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """滑动窗口切分（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    步长 = chunk_size - overlap，切点尽量对齐句末标点，返回非空片段列表"""
    if len(text) <= chunk_size:
        return [text]
    pieces, step, start = [], chunk_size - overlap, 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            end = _find_cut(text, end, chunk_size)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = end - overlap  # 回退 overlap 形成重叠
        if start < 0:
            start = 0
    return pieces


def _split_paragraphs(body_text: str) -> List[str]:
    """正文按段落边界拆分（空行/换行），过滤空段"""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", body_text) if p.strip()]
    return paras


def _is_table_text(t: str) -> bool:
    """含 Markdown 表格分隔线即视为表格（与 tables_structured.markdown 对应）"""
    return "| ---" in t or "| --- |" in t


def build_parent_blocks(parsed_data: Dict[str, Any], parent_max: int = PARENT_MAX_CHARS) -> List[Dict[str, Any]]:
    """语义分块第一层：按标题层级 + 段落边界构建父块（人工智能NLP-RAG-基于PDF文档的问答系统优化）
    父块 = 语义完整单元：heading 开启新父块；正文段落聚合至 parent_max；表格 markdown 独立成块
    """
    doc_id = parsed_data.get("doc_id", "doc")
    layout_blocks = parsed_data.get("layout_blocks", [])
    # 表格以结构化 markdown 为准（Step 2 产物），按页索引
    tables_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for t in parsed_data.get("tables_structured", []):
        tables_by_page.setdefault(t["page"], []).append(t)

    # 页面级内容流：body+heading 按 y 坐标排序，表格块追加页尾（顺序近似）
    pages_flow: Dict[int, List[Dict[str, Any]]] = {}
    for b in layout_blocks:
        if b["type"] in ("header", "footer"):
            continue  # 页眉页脚噪声不入块
        if b["type"] == "table":
            continue  # 表格用结构化版本替换
        pages_flow.setdefault(b["page"], []).append(b)
    all_pages = sorted(set(pages_flow.keys()) | set(tables_by_page.keys()))

    parents: List[Dict[str, Any]] = []
    cur_text_parts: List[str] = []
    cur_heading = ""
    cur_page = None
    sec_idx = 0

    def _flush():
        nonlocal cur_text_parts, sec_idx
        text = "\n".join(cur_text_parts).strip()
        cur_text_parts = []
        if not text:
            return
        sec_idx += 1
        parents.append({
            "chunk_id": f"{doc_id}_parent_{sec_idx:04d}",
            "text": text,
            "char_count": len(text),
            "chunk_type": "table" if _is_table_text(text) else "text",
            "role": "parent",
            "heading": cur_heading,
            "page": cur_page if isinstance(cur_page, int) else cur_page,
        })

    for page_num in all_pages:
        blocks = sorted(pages_flow.get(page_num, []), key=lambda b: (b.get("y0", 0), b.get("y1", 0)))
        for b in blocks:
            if b["type"] == "heading":
                _flush()  # 标题 = 语义边界：先结清当前父块
                cur_heading = b["text"]
                cur_page = page_num
                cur_text_parts = [b["text"]]
                continue
            # 正文：逐段聚合，超出 parent_max 结清
            for para in _split_paragraphs(b["text"]):
                if not cur_text_parts:
                    cur_page = page_num
                joined = len("\n".join(cur_text_parts + [para]))
                if cur_text_parts and joined > parent_max:
                    _flush()
                    cur_text_parts = [para]
                else:
                    cur_text_parts.append(para)
        # 页尾：本页结构化表格各自独立成父块
        for t in tables_by_page.get(page_num, []):
            md_parts = [t["markdown"]]
            if len(t["markdown"]) > parent_max:  # 超长表格按行聚合多父块
                md_parts = sliding_window(t["markdown"], parent_max, overlap=150)
            for md in md_parts:
                _flush_tmp = cur_text_parts
                cur_text_parts = [md]
                parents.append({
                    "chunk_id": f"{doc_id}_parent_{len(parents) + 1:04d}",
                    "text": md, "char_count": len(md), "chunk_type": "table",
                    "role": "parent", "heading": cur_heading, "page": page_num,
                })
                cur_text_parts = _flush_tmp
    _flush()
    # 重排 parent 编号保证连续
    for i, p in enumerate(parents, start=1):
        p["chunk_id"] = f"{doc_id}_parent_{i:04d}"
        p["parent_index"] = i - 1
    return parents


def build_children(parents: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """语义分块第二层：父块 → 滑动窗口子块（检索用，人工智能NLP-RAG-基于PDF文档的问答系统优化）
    子块继承父块的 page/heading/chunk_type，携带 parent_id 指回父块"""
    doc_id_hint = parents[0]["chunk_id"].rsplit("_parent_", 1)[0] if parents else "doc"
    children: List[Dict[str, Any]] = []
    for p in parents:
        pieces = sliding_window(p["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            children.append({
                "chunk_id": f"{p['chunk_id']}_c{i:02d}",
                "parent_id": p["chunk_id"],
                "text": piece,
                "char_count": len(piece),
                "chunk_type": p["chunk_type"],       # text / table（兼容工单一语义）
                "role": "child",
                "page": p["page"],
                "heading": p["heading"],
                "child_index": i,
            })
    for gi, c in enumerate(children):
        c["global_index"] = gi
    return children


def chunk_semantic(parsed_data: Dict[str, Any], chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> Dict[str, Any]:
    """工单二主入口：语义分块 + 父子块（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    parents = build_parent_blocks(parsed_data)
    children = build_children(parents, chunk_size, overlap)
    text_children = sum(1 for c in children if c["chunk_type"] == "text")
    table_children = len(children) - text_children
    return {
        "doc_id": parsed_data.get("doc_id", ""),
        "filename": parsed_data.get("filename", ""),
        "strategy": "semantic_parent_child_v2",
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "total_parent_chunks": len(parents),
        "total_child_chunks": len(children),
        "text_child_chunks": text_children,
        "table_child_chunks": table_children,
        "parent_chunks": parents,   # 生成用父块（LLM 上下文）
        "chunks": children,          # 检索用子块（向量/BM25 入库）
    }


def compare_with_baseline(baseline_path: str, optimized: Dict[str, Any]) -> Dict[str, Any]:
    """对比工单一分块（数量/平均长度）（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    if not os.path.isfile(baseline_path):
        logger.warning(f"工单一分块文件不存在，跳过对比: {baseline_path}")
        return {}
    with open(baseline_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    base_chunks = base.get("chunks", [])
    base_lens = [c.get("char_count", len(c.get("text", ""))) for c in base_chunks]
    child_lens = [c["char_count"] for c in optimized["chunks"]]
    parent_lens = [p["char_count"] for p in optimized["parent_chunks"]]
    return {
        "baseline": {"total": len(base_chunks), "avg_len": round(sum(base_lens) / max(1, len(base_lens)), 1)},
        "optimized_child": {"total": len(child_lens), "avg_len": round(sum(child_lens) / max(1, len(child_lens)), 1)},
        "optimized_parent": {"total": len(parent_lens), "avg_len": round(sum(parent_lens) / max(1, len(parent_lens)), 1)},
    }


def print_summary(result: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """输出对比统计与示例父子块（工单二验证要求）"""
    print("=" * 64)
    print("工单二语义分块统计（人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    print(f"  策略           : {result['strategy']}")
    print(f"  滑动窗口       : size={result['chunk_size']}, overlap={result['chunk_overlap']}")
    print(f"  父块数量       : {result['total_parent_chunks']}")
    print(f"  子块数量       : {result['total_child_chunks']} "
          f"(text={result['text_child_chunks']}, table={result['table_child_chunks']})")
    print("-" * 64)
    if stats:
        print(f"  工单一基线分块   : {stats['baseline']['total']} 块, 平均 {stats['baseline']['avg_len']} 字")
        print(f"  工单二子块(检索) : {stats['optimized_child']['total']} 块, 平均 {stats['optimized_child']['avg_len']} 字")
        print(f"  工单二父块(生成) : {stats['optimized_parent']['total']} 块, 平均 {stats['optimized_parent']['avg_len']} 字")
        print("-" * 64)
    # 示例父子块：找一个有多个子块的父块展示
    demo_parent = None
    for p in result["parent_chunks"]:
        kids = [c for c in result["chunks"] if c["parent_id"] == p["chunk_id"]]
        if len(kids) >= 2 and p["chunk_type"] == "text":
            demo_parent = (p, kids)
            break
    if demo_parent is None and result["parent_chunks"]:
        p = result["parent_chunks"][0]
        demo_parent = (p, [c for c in result["chunks"] if c["parent_id"] == p["chunk_id"]][:1])
    if demo_parent:
        p, kids = demo_parent
        print(f"[父块示例] {p['chunk_id']} (page={p['page']}, {p['chunk_type']}, {p['char_count']}字)")
        print(f"  heading: {p['heading']}")
        print(f"  text[:120]: {p['text'][:120]}...")
        for k in kids[:2]:
            print(f"  [子块] {k['chunk_id']} ({k['char_count']}字) text[:80]: {k['text'][:80]}...")


def main(parsed_path: str = DEFAULT_PARSED, out_path: str = DEFAULT_OUT,
         baseline_path: str = DEFAULT_BASELINE_CHUNKS, chunk_size: int = CHUNK_SIZE,
         overlap: int = CHUNK_OVERLAP) -> Dict[str, Any]:
    """工单二默认流程（python -m src.chunker_optimized）"""
    with open(parsed_path, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)
    result = chunk_semantic(parsed_data, chunk_size, overlap)
    stats = compare_with_baseline(baseline_path, result)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"优化分块已写入: {out_path}")
    print_summary(result, stats)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="工单二语义分块器（人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    ap.add_argument("--parsed", default=DEFAULT_PARSED, help="Step 2 优化版解析 JSON")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出分块 JSON")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE_CHUNKS, help="工单一分块 JSON（对比用）")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = ap.parse_args()
    main(args.parsed, args.out, args.baseline, args.chunk_size, args.overlap)
