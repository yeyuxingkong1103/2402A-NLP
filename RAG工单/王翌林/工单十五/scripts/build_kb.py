# -*- coding: utf-8 -*-
"""
工单十五：知识库构建脚本
流程：PDF高清渲染 → Qwen2-VL OCR → 分块 → bge-m3向量化 → Milvus入库
      → 图纸结构化描述（图号↔描述绑定）
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
from src.image_knowledge import ImageKnowledgeBase


def build_knowledge_base(pdf_path: str, data_dir: str = "./data", doc_id: str = "CN100347506C"):
    filename = os.path.basename(pdf_path)
    pages_dir = os.path.join(data_dir, "pages_hd")
    ocr_dir = os.path.join(data_dir, "ocr")
    chunks_dir = os.path.join(data_dir, "chunks")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    logger.info("=== Step 1: 提取 PDF 页面高清图片 ===")
    page_info = extract_images_from_pdf(pdf_path, pages_dir, zoom=4.0)
    page_images = {p["page"]: get_best_image(p) for p in page_info["pages"]}

    logger.info("=== Step 2: Qwen2-VL OCR ===")
    ocr_json = os.path.join(ocr_dir, "all_pages_ocr.json")
    if os.path.isfile(ocr_json):
        logger.info(f"复用已有 OCR 结果: {ocr_json}")
        with open(ocr_json, encoding="utf-8") as f:
            ocr_results = json.load(f)
    else:
        ocr_results = []
        for page_num in sorted(page_images.keys()):
            text = vlm_engine.ocr_image(page_images[page_num])
            ocr_results.append({"page": page_num, "image_path": page_images[page_num], "text": text})
            logger.info(f"第{page_num}页 OCR: {len(text)}字")
        with open(ocr_json, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)

    logger.info("=== Step 3: 文本分块 ===")
    chunk_result = chunk_ocr_results(ocr_results, doc_id, filename)
    chunks_json = os.path.join(chunks_dir, f"{doc_id}_chunks.json")
    with open(chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunk_result, f, ensure_ascii=False, indent=2)
    chunks = chunk_result["chunks"]
    logger.info(f"分块完成: {len(chunks)} chunks")

    logger.info("=== Step 4: 向量化 + Milvus 入库 ===")
    embedder = get_embedder()
    vectors = embedder.encode([c["text"] for c in chunks], show_progress_bar=True)
    vs = VectorStore()
    vs.drop_collection()
    vs.insert_chunks(chunks, vectors)
    stats = vs.get_stats()
    logger.info(f"入库完成: {stats}")

    logger.info("=== Step 5: 保存图纸知识库 + 页面映射 ===")
    img_kb = ImageKnowledgeBase()
    img_kb.save(os.path.join(data_dir, "image_knowledge.json"))

    mapping = {"doc_id": doc_id, "filename": filename,
               "page_images": {str(k): v for k, v in page_images.items()},
               "page_texts": {str(r["page"]): r["text"] for r in ocr_results},
               "chunks_json": chunks_json, "ocr_json": ocr_json,
               "figure_knowledge": img_kb.kb}
    mapping_json = os.path.join(data_dir, "kb_mapping.json")
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info(f"=== 知识库构建完成 ===")
    return mapping


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=f"{WORK_ORDER_NO} 知识库构建")
    p.add_argument("--pdf", default="./附件/CN100347506C.pdf")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--doc-id", default="CN100347506C")
    args = p.parse_args()
    build_knowledge_base(args.pdf, args.data_dir, args.doc_id)
