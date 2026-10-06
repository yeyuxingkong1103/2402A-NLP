# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
独立入库脚本（优化版）：解析 PDF → 切分(400/50) → 入库(批25) → 重建 BM25。

用法：
    cd src
    python ingest_pdf.py            # 处理 data 目录下所有 PDF
    python ingest_pdf.py file1.pdf  # 处理指定文件
"""

import os
import sys

from pdf_parser import parse_pdf_to_document
from text_splitter import split_documents
from db_milvus import ingest_documents, get_collection_count, get_milvus_client
from retriever import refresh_bm25, clear_cache
from logger import get_logger

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    if len(sys.argv) > 1:
        pdf_files = [os.path.join(DATA_DIR, n) for n in sys.argv[1:]]
    else:
        pdf_files = [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.lower().endswith(".pdf")
        ]

    if not pdf_files:
        logger.error(f"data 目录下没有 PDF：{DATA_DIR}")
        sys.exit(1)

    logger.info(f"开始入库：共 {len(pdf_files)} 个 PDF（优化版 chunk_size=400）")
    ingested = 0
    for i, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"[{i}/{len(pdf_files)}] 处理：{os.path.basename(pdf_path)}")
        if not os.path.isfile(pdf_path):
            logger.error(f"文件不存在：{pdf_path}")
            continue
        try:
            docs = parse_pdf_to_document(pdf_path)
            if not docs:
                continue
            chunks = split_documents(docs)
            if not chunks:
                continue
            ingest_documents(chunks)
            ingested += 1
        except Exception as e:
            logger.error(f"处理失败：{e}")

    refresh_bm25()
    clear_cache()
    count = get_collection_count()
    logger.info(f"入库完成：成功 {ingested}/{len(pdf_files)}，集合总数 {count}")


if __name__ == "__main__":
    main()
