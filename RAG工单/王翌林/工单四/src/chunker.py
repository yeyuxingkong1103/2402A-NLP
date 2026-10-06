# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/chunker.py — LangChain RecursiveCharacterTextSplitter 分块（优化版 v2）

优化项：
  1. 分块参数调优：chunk_size 800→600, overlap 150→120（更细粒度，更高检索精度）
  2. 表格感知：表格内容用更小 chunk_size=400 保持完整性
  3. 中文分隔符增强：引号、顿号、冒号、书名号等
"""
import argparse, json, os, re
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 优化后默认参数
DEFAULT_CHUNK_SIZE = 600       # 原 800 → 更细粒度，提高检索精度
DEFAULT_CHUNK_OVERLAP = 120    # 原 150 → 匹配更短 size，节省 token
TABLE_CHUNK_SIZE = 400         # 表格专用更小 size 保持完整性
TABLE_CHUNK_OVERLAP = 80

# 工单：中文分隔符增强（加引号、顿号、冒号、书名号等）
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、",
                       "”", "）", "】", "》", " ", ""]
TABLE_SEPARATORS = ["\n\n", "\n", "|", "  ", "	", ""]  # 表格优先按换行和竖线切

def build_splitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, separators=None):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=separators or DEFAULT_SEPARATORS, length_function=len, is_separator_regex=False)

# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 表格内容检测
def _is_table_like(text: str) -> bool:
    """检测文本是否像表格（含多行数字/短文本 + 可能的竖线分隔）"""
    if not text or len(text) < 50: return False
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3: return False
    # 竖线表格
    if sum(1 for l in lines if "|" in l) >= len(lines) * 0.5: return True
    # 多数字行
    digit_lines = sum(1 for l in lines if re.search(r'\d', l))
    if digit_lines >= len(lines) * 0.6: return True
    # 等宽空格对齐
    if sum(1 for l in lines if "  " in l) >= len(lines) * 0.4: return True
    return False

def chunk_parsed_document(parsed_data: Dict[str, Any], chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                          use_table_aware=True) -> Dict[str, Any]:
    """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 优化版分块"""
    normal_splitter = build_splitter(chunk_size, chunk_overlap)
    table_splitter = build_splitter(TABLE_CHUNK_SIZE, TABLE_CHUNK_OVERLAP, TABLE_SEPARATORS)
    doc_id = parsed_data.get("doc_id", "")
    filename = parsed_data.get("filename", "")
    pages = parsed_data.get("pages", [])
    all_chunks, global_idx = [], 0
    for page_item in pages:
        page_num = page_item.get("page", 0)
        page_text = page_item.get("text", "") or ""
        if not page_text.strip(): continue
        # 工单：表格感知分块
        if use_table_aware and _is_table_like(page_text):
            splitter = table_splitter
            chunk_type = "table"
        else:
            splitter = normal_splitter
            chunk_type = "text"
        splits = splitter.split_text(page_text)
        for local_i, text in enumerate(splits):
            text = text.strip()
            if not text: continue
            all_chunks.append({
                "chunk_id": f"{doc_id}_p{page_num}_c{local_i}",
                "page": page_num,
                "chunk_index": local_i,
                "global_index": global_idx,
                "text": text,
                "char_count": len(text),
                "chunk_type": chunk_type,  # 工单：标记 chunk 类型（text/table）
                "source": f"{filename}#page={page_num}",
            })
            global_idx += 1
    table_count = sum(1 for c in all_chunks if c.get("chunk_type") == "table")
    text_count = len(all_chunks) - table_count
    logger.info(f"分块完成: chunks={len(all_chunks)} (text={text_count}, table={table_count}), size={chunk_size}, overlap={chunk_overlap}")
    return {"doc_id": doc_id, "filename": filename, "total_pages": parsed_data.get("total_pages", len(pages)),
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
            "total_chunks": len(all_chunks), "table_chunks": table_count, "text_chunks": text_count,
            "chunks": all_chunks}

def chunk_file(parsed_path: str, out_path: str, **kwargs) -> str:
    with open(parsed_path, "r", encoding="utf-8") as f: parsed_data = json.load(f)
    result = chunk_parsed_document(parsed_data, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Chunks 已写入: {out_path}"); return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PDF 分块器（工单：人工智能NLP-RAG-基于PDF文档的问答系统）")
    p.add_argument("--parsed", default="data/parsed/招股说明书1.json")
    p.add_argument("--out", default=None)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = p.parse_args()
    out_path = args.out or f"data/chunks/{Path(args.parsed).stem}_chunks.json"
    chunk_file(args.parsed, out_path, chunk_size=args.chunk_size, chunk_overlap=args.overlap)
