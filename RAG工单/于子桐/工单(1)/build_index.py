# -*- coding: utf-8 -*-
"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
功能: 解析《招股说明书1.pdf》-> 文本分块 -> bge-m3 向量化 -> 落盘索引

运行: python build_index.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_common as R

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "附件", "招股说明书1.pdf")
INDEX_DIR = os.path.join(HERE, "index", "doc1_text")

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("工单01 / 工单02 —— 构建《招股说明书1.pdf》向量索引")
    print("=" * 60)
    if R.VectorStore.exists(INDEX_DIR):
        print("索引已存在, 跳过。如需重建请删除:", INDEX_DIR)
        sys.exit(0)

    # 工单01 基线方案: 纯文本解析, 500 字分块 / 50 字重叠
    vs = R.build_index([PDF], INDEX_DIR, chunk_size=500, overlap=50,
                       with_tables=False, with_images=False)
    print(f"完成, 共 {len(vs.chunks)} 块, 耗时 {time.time()-t0:.1f}s")
    print("索引目录:", INDEX_DIR)
