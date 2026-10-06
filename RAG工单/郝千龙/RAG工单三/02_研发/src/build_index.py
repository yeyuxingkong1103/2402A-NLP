# -*- coding: utf-8 -*-
# 【索引构建入口 · build_index.py】解析双招股书 → 表格感知分块 → 构建TF-IDF/BM25双路索引并持久化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""离线构建：遍历 CONFIG.pdf_docs 中的两本招股书 PDF，解析、分块、建库，
合并写入同一个 index_store；加 --baseline 构建“无表格朴素链路”基线索引
（定长滑窗、不启用 pdfplumber 表格解析），供优化前后对比评测使用。

用法（PowerShell）：
    python build_index.py            # 优化链路
    python build_index.py --baseline # 基线链路
"""
import argparse
import os
import sys
import time

from config import CONFIG
from pdf_parser import parse_pdf
from chunker import build_chunks, build_naive_chunks
from vector_store import IndexStore, create_embedder


def build(baseline: bool = False) -> None:
    """执行一次完整索引构建。

    :param baseline: True=基线（无表格+定长滑窗）；False=优化（表格感知）
    """
    tag = "基线（无表格·定长滑窗）" if baseline else "优化（表格感知）"
    index_dir = CONFIG.index_dir + ("_baseline" if baseline else "")
    print(f"[建库] 模式：{tag}")
    print(f"[建库] 索引目录：{index_dir}")

    all_blocks = []
    for doc in CONFIG.pdf_docs:
        if not os.path.exists(doc["file"]):
            print(f"[建库][错误] 找不到PDF：{doc['file']}")
            sys.exit(1)
        t0 = time.perf_counter()
        print(f"[建库] 解析 {os.path.basename(doc['file'])}"
              f"（{doc['short']}，表格解析={'关闭' if baseline else '开启'}）…")
        blocks = parse_pdf(
            doc["file"], doc["company"], enable_tables=not baseline)
        n_table = sum(1 for b in blocks if b.type == "table")
        print(f"[建库]   得到 {len(blocks)} 个Block（其中表格 {n_table} 张），"
              f"耗时 {time.perf_counter() - t0:.1f} 秒")
        all_blocks.extend(blocks)

    t1 = time.perf_counter()
    if baseline:
        chunks = build_naive_chunks(all_blocks)
    else:
        chunks = build_chunks(all_blocks)
    print(f"[分块] 共 {len(chunks)} 个检索块，耗时 "
          f"{time.perf_counter() - t1:.1f} 秒（"
          f"整表块 {sum(1 for c in chunks if c.chunk_type == 'table')}，"
          f"行块 {sum(1 for c in chunks if c.chunk_type == 'table_row')}，"
          f"正文块 {sum(1 for c in chunks if c.chunk_type == 'text')}）")

    t2 = time.perf_counter()
    embedder = create_embedder([c.text for c in chunks])
    store = IndexStore.build(chunks, embedder)
    store.save(index_dir)
    print(f"[建库] 索引已保存，耗时 {time.perf_counter() - t2:.1f} 秒")
    print("[建库] 完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建招股书RAG索引")
    parser.add_argument("--baseline", action="store_true",
                        help="构建无表格朴素基线索引")
    args = parser.parse_args()
    build(baseline=args.baseline)
