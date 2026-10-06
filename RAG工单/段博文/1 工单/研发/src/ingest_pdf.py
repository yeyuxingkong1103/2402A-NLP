# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
PDF 入库脚本（可独立运行）。

完整链路：
    扫描 data 目录下所有 .pdf 文件 →
    pdf_parser.parse_pdf_to_document 按页提取 →
    text_splitter.split_documents 切分 →
    db_milvus.ingest_documents 向量化写入 Milvus →
    retriever.refresh_bm25 重建关键词索引。

用法：
    python ingest_pdf.py           # 处理 data 目录下所有 PDF
    python ingest_pdf.py a.pdf      # 只处理指定文件（相对 data 目录）

设计取舍：
    1. 入库失败单个文件不中断整批：每份 PDF 用 try 包住，失败只记 error，
       继续处理下一份；
    2. 切分用 split_documents（保留页码 metadata），而不是 split_text，
       这样检索命中后能向前端展示「出自第几页」；
    3. 入库完一次性 rebuild BM25，而不是每份 PDF 后都重建（BM25 重建是
       全量操作，多次重建浪费时间）。
"""

import os  # 路径操作
import sys  # 命令行参数

from pdf_parser import parse_pdf_to_document  # PDF 解析
from text_splitter import split_documents  # 文本切分
from db_milvus import ingest_documents, get_collection_count  # Milvus 入库
from retriever import refresh_bm25  # BM25 重建
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# data 目录：脚本所在目录下的 data 文件夹
# 用 __file__ 拼绝对路径，任意工作目录启动都能找到 data 目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def ingest_pdf_file(file_path: str) -> int:
    """
    处理单个 PDF 文件：解析 → 切分 → 入库

    参数：
        file_path：PDF 文件绝对路径。
    返回：
        int，本次入库后集合内语料总数；解析失败/切分失败/入库失败时返回 0。
    异常与降级：
        本函数内部 try 包住所有异常：解析失败、切分失败、入库失败都只记
        error 日志并返回 0，不抛异常——批量入库时不应该因为单个文件
        损坏就中断整批流程。
    """
    try:
        # 1. 解析
        docs = parse_pdf_to_document(file_path)  # 按页提取 Document 列表
        if not docs:
            logger.warning(f"文件 {file_path} 解析为空，跳过")
            return 0

        # 2. 切分
        chunks = split_documents(docs)  # 切成更小的 chunk，保留页码 metadata
        if not chunks:
            logger.warning(f"文件 {file_path} 切分为空，跳过")
            return 0

        # 3. 入库（内部已做幂等去重，重复入库同一文件不会产生重复向量）
        total = ingest_documents(chunks)
        logger.info(f"文件 {file_path} 入库完成：本批 chunk={len(chunks)}，集合总数={total}")
        return total
    except Exception as e:
        logger.error(f"处理文件 {file_path} 失败：{e}")
        return 0


def main():
    """
    脚本入口：扫描 data 目录下所有 PDF 并入库，最后重建 BM25 索引

    命令行参数：
        不传：处理 data 目录下所有 .pdf
        传若干文件名（相对 data 目录）：只处理这些文件
    """
    # 1. 确定要处理的文件列表
    if not os.path.isdir(DATA_DIR):  # data 目录不存在
        logger.error(f"data 目录不存在：{DATA_DIR}，请先创建并放入 PDF 文件")
        return

    if len(sys.argv) > 1:  # 命令行传了文件名
        pdf_files = [os.path.join(DATA_DIR, name) for name in sys.argv[1:]]
    else:  # 扫描全部
        pdf_files = [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.lower().endswith(".pdf")
        ]

    if not pdf_files:
        logger.warning(f"data 目录下没有 PDF 文件：{DATA_DIR}")
        return

    logger.info(f"开始入库：共 {len(pdf_files)} 个 PDF 文件")

    # 2. 逐文件处理（单文件失败不中断整批）
    success = 0  # 成功计数
    for i, pdf in enumerate(pdf_files, start=1):
        logger.info(f"[{i}/{len(pdf_files)}] 处理：{os.path.basename(pdf)}")
        total = ingest_pdf_file(pdf)
        if total > 0 or os.path.isfile(pdf):  # 文件存在就算尝试过
            success += 1 if total > 0 else 0

    logger.info(f"入库完成：成功 {success}/{len(pdf_files)} 个文件")

    # 3. 重建 BM25 索引（所有文件入库完才重建，避免重复全量重建）
    try:
        refresh_bm25()
    except Exception as e:
        logger.error(f"重建 BM25 失败：{e}")

    # 4. 打印最终状态
    count = get_collection_count()
    logger.info(f"集合当前语料数：{count}")


if __name__ == "__main__":
    main()
