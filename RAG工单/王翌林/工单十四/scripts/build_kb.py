# -*- coding: utf-8 -*-
"""
工单十四：知识库构建脚本
流程：PDF图片提取 → Qwen2-VL OCR → 分块 → bge-m3向量化 → Milvus入库
"""
import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import WORK_ORDER_NO
from src.pdf_parser import extract_images_from_pdf, get_best_image
from src import vlm_engine
from src.chunker import chunk_ocr_results
from src.embedding import get_embedder
from src.vector_store import VectorStore


def build_knowledge_base(pdf_path: str, data_dir: str = "./data", doc_id: str = "CN100342976C"):
    filename = os.path.basename(pdf_path)
    pages_dir = os.path.join(data_dir, "pages")
    ocr_dir = os.path.join(data_dir, "ocr")
    chunks_dir = os.path.join(data_dir, "chunks")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    # Step 1: 提取图片
    logger.info("=== Step 1: 提取 PDF 页面图片 ===")
    page_info = extract_images_from_pdf(pdf_path, pages_dir, zoom=4.0)
    page_images = {p["page"]: get_best_image(p) for p in page_info["pages"]}

    # Step 2: OCR
    logger.info("=== Step 2: Qwen2-VL OCR ===")
    ocr_results = []
    for page_num in sorted(page_images.keys()):
        img_path = page_images[page_num]
        text = vlm_engine.ocr_image(img_path)
        ocr_results.append({"page": page_num, "image_path": img_path, "text": text})
        logger.info(f"第{page_num}页 OCR: {len(text)}字")
    ocr_json = os.path.join(ocr_dir, "all_pages_ocr.json")
    with open(ocr_json, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)

    # Step 3: 分块
    logger.info("=== Step 3: 文本分块 ===")
    chunk_result = chunk_ocr_results(ocr_results, doc_id, filename)
    chunks_json = os.path.join(chunks_dir, f"{doc_id}_chunks.json")
    with open(chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunk_result, f, ensure_ascii=False, indent=2)
    chunks = chunk_result["chunks"]

    # Step 4: 向量化 + 入库
    logger.info("=== Step 4: 向量化 + Milvus 入库 ===")
    embedder = get_embedder()
    vectors = embedder.encode([c["text"] for c in chunks], show_progress_bar=True)
    vs = VectorStore()
    vs.drop_collection()
    vs.insert_chunks(chunks, vectors)
    stats = vs.get_stats()
    logger.info(f"入库完成: {stats}")

    # 保存页面映射供 RAG 引擎使用
    mapping = {"doc_id": doc_id, "filename": filename,
               "page_images": {str(k): v for k, v in page_images.items()},
               "page_texts": {str(r["page"]): r["text"] for r in ocr_results},
               "chunks_json": chunks_json, "ocr_json": ocr_json}
    mapping_json = os.path.join(data_dir, "kb_mapping.json")
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    logger.info(f"=== 知识库构建完成 ===")
    return mapping


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=f"{WORK_ORDER_NO} 知识库构建")
    p.add_argument("--pdf", default="./附件/CN100342976C.pdf")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--doc-id", default="CN100342976C")
    args = p.parse_args()
    build_knowledge_base(args.pdf, args.data_dir, args.doc_id)
