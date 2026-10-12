# -*- coding: utf-8 -*-
"""
工单十四：文本分块器
针对专利文档（OCR文本），按页分块，保留页码溯源。
"""
import json
import os
from typing import Any, Dict, List
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、", "”", "）", "】", "》", " ", ""]


def build_splitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS, length_function=len)


def chunk_ocr_results(ocr_results: List[Dict], doc_id: str, filename: str,
                      chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP) -> Dict[str, Any]:
    """
    将 OCR 结果（每页文本）分块，保留 page 溯源。
    ocr_results: [{"page": 1, "text": "...", ...}, ...]
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    all_chunks = []
    global_idx = 0
    for page_item in ocr_results:
        page_num = page_item.get("page", 0)
        page_text = page_item.get("text", "") or ""
        if not page_text.strip():
            continue
        splits = splitter.split_text(page_text)
        for local_i, text in enumerate(splits):
            text = text.strip()
            if not text:
                continue
            all_chunks.append({
                "chunk_id": f"{doc_id}_p{page_num}_c{local_i}",
                "page": page_num,
                "chunk_index": local_i,
                "global_index": global_idx,
                "text": text,
                "char_count": len(text),
                "source": f"{filename}#page={page_num}",
            })
            global_idx += 1
    logger.info(f"分块完成: {len(all_chunks)} chunks, size={chunk_size}, overlap={chunk_overlap}")
    return {"doc_id": doc_id, "filename": filename,
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
            "total_chunks": len(all_chunks), "chunks": all_chunks}


def chunk_file(ocr_json_path: str, out_path: str, doc_id: str = "CN100342976C",
               filename: str = "CN100342976C.pdf", **kwargs) -> str:
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_results = json.load(f)
    result = chunk_ocr_results(ocr_results, doc_id, filename, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Chunks 已写入: {out_path}")
    return out_path
