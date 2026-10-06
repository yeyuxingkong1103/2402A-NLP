# -*- coding: utf-8 -*-
# 【建库入口 · build_index.py】解析两份招股书→抽图/OCR→图文统一分块→双索引落盘
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""建库命令行（在02_研发/src目录下执行）：

    python build_index.py

流程：打开两份PDF清单 → 文本/表格解析 → 嵌入图片抽取+矢量图渲染+离线OCR
     → 图文统一分块 → BGE/TF-IDF向量+BM25双索引 → 持久化。
OCR为CPU离线批处理（单图数秒），结果增量缓存，仅建库期执行，不计入问答耗时。
"""
import json
import logging
import os
import sys
import time

# 本机实测：PaddlePaddle先于torch加载会造成OpenMP/MKL运行时冲突，
# BGE首次推理零CPU挂死。必须在导入paddle相关模块前预加载torch。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch  # noqa: F401,E402
from sentence_transformers import SentenceTransformer  # noqa: F401,E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from image_extractor import OcrEngine, extract_figures
from multimodal_index import IndexStore, build_chunks, create_embedder
from pdf_parser import PDFParseError, PdfInventory, parse_inventory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build() -> None:
    """执行完整建库流程并落盘索引与图像库。"""
    t0 = time.perf_counter()
    all_blocks = []
    all_figures = []
    doc_stats = {}

    logger.info("① 初始化离线OCR引擎（PaddleOCR不可用时自动降级）")
    ocr = OcrEngine()
    cache_path = os.path.join(CONFIG.image_dir, CONFIG.ocr_cache_name)
    os.makedirs(CONFIG.image_dir, exist_ok=True)
    ocr_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)
        logger.info("已载入OCR缓存 %d 条", len(ocr_cache))

    for doc_tag, filename in CONFIG.pdf_docs.items():
        pdf_path = os.path.join(CONFIG.root_dir, filename)
        logger.info("② 解析PDF：%s（%s）", filename, doc_tag)
        try:
            inv = PdfInventory(pdf_path, doc_tag)
        except PDFParseError as exc:
            logger.error("文档解析失败，已中止建库：%s", exc)
            raise
        with inv:
            page_count = inv.page_count
            blocks = parse_inventory(inv)
            logger.info("  文本/表格块 %d 个，开始图像抽取", len(blocks))
            figures = extract_figures(inv, CONFIG.image_dir, ocr, ocr_cache)
            # 图像抽取完即落盘OCR缓存，支持断点续建
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(ocr_cache, f, ensure_ascii=False)
        raster_n = sum(1 for f in figures if f.kind == "raster")
        vector_n = sum(1 for f in figures if f.kind == "vector")
        doc_stats[doc_tag] = {"pages": page_count, "blocks": len(blocks),
                              "images_raster": raster_n,
                              "images_vector": vector_n,
                              "ocr_available": ocr.available}
        logger.info("  %s：嵌入位图%d张，矢量渲染图%d张",
                    doc_tag, raster_n, vector_n)
        all_blocks.extend(blocks)
        all_figures.extend(figures)

    logger.info("③ 图文统一分块")
    chunks = build_chunks(all_blocks, all_figures)
    type_count = {"text": 0, "table": 0, "image": 0}
    for c in chunks:
        type_count[c.chunk_type] = type_count.get(c.chunk_type, 0) + 1
    logger.info("  总块数 %d（文本%d/表格%d/图像%d）",
                len(chunks), type_count["text"], type_count["table"],
                type_count["image"])

    logger.info("④ 加载嵌入模型并编码（本地BGE优先，否则TF-IDF）")
    embedder = create_embedder([c.text for c in chunks])

    logger.info("⑤ 构建稠密向量+BM25双索引并持久化")
    store = IndexStore.build(chunks, embedder)
    stats = {"documents": doc_stats,
             "ocr_engine": "paddleocr(PP-OCRv6)" if ocr.available else "degraded",
             "clip_encoder": "disabled(no_local_cache)",
             "type_count": type_count}
    store.save(CONFIG.index_dir, stats=stats)
    logger.info("建库完成，总耗时 %.1f s，索引目录：%s",
                time.perf_counter() - t0, CONFIG.index_dir)


if __name__ == "__main__":
    build()
