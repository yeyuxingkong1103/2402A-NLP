# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
批量入库脚本：解析 ccf_competition 中的 9 份年报 PDF 并写入 Milvus。

流程：
    读取 PDF → 解析文本/表格 → 切分(400/50) → 向量化 → 写入 Milvus
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pdf_parser import parse_pdf_to_document
from text_splitter import split_documents
from db_milvus import ingest_documents, get_collection_count, clear_collection

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ccf_competition", "pdf")


def main():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    print(f"发现 {len(pdf_files)} 个 PDF 文件")

    # 清空旧集合
    print("清空旧集合...")
    clear_collection()

    total_chunks = 0
    for i, fname in enumerate(pdf_files, 1):
        path = os.path.join(DATA_DIR, fname)
        print(f"\n[{i}/{len(pdf_files)}] {fname}")

        try:
            docs = parse_pdf_to_document(path)
            if not docs:
                print(f"  解析为空，跳过")
                continue

            chunks = split_documents(docs)
            if not chunks:
                print(f"  切分为空，跳过")
                continue

            ingest_documents(chunks)
            total_chunks += len(chunks)
            print(f"  入库 {len(chunks)} 块")
        except Exception as e:
            print(f"  失败: {e}")

    count = get_collection_count()
    print(f"\n{'='*40}")
    print(f"入库完成: 成功 {len(pdf_files)}/{len(pdf_files)} 个文件")
    print(f"总块数: {total_chunks}")
    print(f"集合总数: {count}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
