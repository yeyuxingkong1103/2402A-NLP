# -*- coding: utf-8 -*-
"""
工单十四：从已有OCR结果构建知识库（分块 + 向量化 + 入库）
不需要重新OCR，直接读取 data/ocr/all_pages_ocr_vlm.json
"""
import os
import sys
import json
import argparse
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chunker import chunk_ocr_results
from src.embedding import get_embedder
from src.vector_store import VectorStore
from src.pdf_parser import extract_images_from_pdf, get_best_image


def build_from_ocr(pdf_path: str, ocr_json: str, data_dir: str = "./data",
                   doc_id: str = "CN100342976C"):
    filename = os.path.basename(pdf_path)
    pages_dir = os.path.join(data_dir, "pages")
    chunks_dir = os.path.join(data_dir, "chunks")

    # 加载 OCR 结果
    with open(ocr_json, "r", encoding="utf-8") as f:
        ocr_results = json.load(f)
    logger.info(f"加载 OCR 结果: {len(ocr_results)} 页")

    # 提取页面图片（供 VQA 使用）
    page_info = extract_images_from_pdf(pdf_path, pages_dir, zoom=4.0)
    page_images = {p["page"]: get_best_image(p) for p in page_info["pages"]}

    # 分块
    chunk_result = chunk_ocr_results(ocr_results, doc_id, filename)
    chunks_json = os.path.join(chunks_dir, f"{doc_id}_chunks.json")
    os.makedirs(chunks_dir, exist_ok=True)
    with open(chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunk_result, f, ensure_ascii=False, indent=2)
    chunks = chunk_result["chunks"]
    logger.info(f"分块完成: {len(chunks)} chunks")

    # 向量化 + 入库
    embedder = get_embedder()
    vectors = embedder.encode([c["text"] for c in chunks], show_progress_bar=True)
    vs = VectorStore()
    vs.drop_collection()
    vs.insert_chunks(chunks, vectors)
    stats = vs.get_stats()
    logger.info(f"入库完成: {stats}")

    # 保存映射
    mapping = {
        "doc_id": doc_id, "filename": filename,
        "page_images": {str(k): v for k, v in page_images.items()},
        "page_texts": {str(r["page"]): r["text"] for r in ocr_results},
        "chunks_json": chunks_json, "ocr_json": ocr_json,
        "total_chunks": len(chunks),
    }
    mapping_json = os.path.join(data_dir, "kb_mapping.json")
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info(f"知识库映射已保存: {mapping_json}")
    return mapping


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default="./附件/CN100342976C.pdf")
    p.add_argument("--ocr", default="./data/ocr/all_pages_ocr_vlm.json")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--doc-id", default="CN100342976C")
    args = p.parse_args()
    build_from_ocr(args.pdf, args.ocr, args.data_dir, args.doc_id)
