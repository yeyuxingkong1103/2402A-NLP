# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
数据入库脚本：批量处理 PDF 文件，解析（文本+表格+图像语义）并写入 Milvus。
"""

import os
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from logger import get_logger
from db_milvus import ingest_documents, get_collection_count, clear_collection
from pdf_parser import parse_pdf_to_document
from text_splitter import split_documents
from retriever import refresh_bm25, clear_cache
from llm_client import clear_llm_cache

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    """主函数：批量入库 PDF 文件。"""
    logger.info("=" * 60)
    logger.info("开始数据入库（图像内容解析增强版）")
    logger.info("=" * 60)

    # 获取 PDF 文件列表
    if not os.path.isdir(DATA_DIR):
        logger.error(f"data 目录不存在：{DATA_DIR}")
        return

    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        logger.error("data 目录下没有 PDF 文件")
        return

    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
    for f in pdf_files:
        logger.info(f"  - {os.path.basename(f)}")

    # 询问是否清空现有集合
    count = get_collection_count()
    if count > 0:
        logger.warning(f"集合中已有 {count} 条数据")
        response = input("是否清空现有集合？(y/n): ")
        if response.lower() == "y":
            clear_collection()
            logger.info("集合已清空")

    # 批量处理
    total_chunks = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n[{i}/{len(pdf_files)}] 处理：{os.path.basename(pdf_path)}")

        try:
            # 1. 解析 PDF
            logger.info("  步骤 1: 解析 PDF（文本 + 表格 + 图像语义）...")
            docs = parse_pdf_to_document(pdf_path)
            if not docs:
                logger.warning(f"  解析为空，跳过")
                continue
            logger.info(f"  解析完成：{len(docs)} 个块")

            # 2. 切分
            logger.info("  步骤 2: 切分文档...")
            chunks = split_documents(docs)
            logger.info(f"  切分完成：{len(chunks)} 个块")

            # 3. 入库
            logger.info("  步骤 3: 写入 Milvus...")
            ingest_documents(chunks)
            total_chunks += len(chunks)

        except Exception as e:
            logger.error(f"  处理失败：{e}")
            continue

    # 重建索引
    logger.info("\n重建 BM25 索引...")
    try:
        refresh_bm25()
        logger.info("BM25 索引重建完成")
    except Exception as e:
        logger.error(f"BM25 索引重建失败：{e}")

    # 清空缓存
    logger.info("清空缓存...")
    clear_cache()
    clear_llm_cache()

    # 统计
    final_count = get_collection_count()
    logger.info("\n" + "=" * 60)
    logger.info("入库完成！")
    logger.info(f"  处理文件数：{len(pdf_files)}")
    logger.info(f"  总块数：{total_chunks}")
    logger.info(f"  集合总数：{final_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
