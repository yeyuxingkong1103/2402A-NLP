# -*- coding: utf-8 -*-
"""知识库构建入口（图像内容解析与检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

一键完成：
1. 抽取并解析 PDF 图形（组织结构图 / 统计图）→ 图像语义块 + CLIP 图像向量库；
2. 构建【优化后】文本向量库（正文 + 结构化表格 + 图像语义块）；
3. 构建【优化前】基线向量库（表格拍平、无图像解析），用于 before/after 对比。

用法：
    python build_kb.py            # 构建（已存在则跳过）
    python build_kb.py --rebuild  # 强制重建
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.knowledge_base import build_kb, build_documents, build_baseline_documents, kb_stats
from src.image_index import get_image_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()

    t0 = time.time()
    print("=" * 68)
    print("【1/3】构建【优化后】文本向量库（正文 + 表格 + 图像语义块）")
    print("=" * 68)
    docs = build_documents()
    print(f"  文档块总数: {len(docs)}")
    for t in ("text", "table", "table_kv", "figure"):
        print(f"    - {t:9s}: {sum(1 for d in docs if d.metadata.get('ctype') == t)}")
    build_kb(docs=docs, persist_directory=config.DB_DIR, force_rebuild=args.rebuild)
    print("  优化后向量库:", kb_stats(config.DB_DIR))

    print("=" * 68)
    print("【2/3】构建【优化前】基线向量库（表格拍平，无图像解析）")
    print("=" * 68)
    base_docs = build_baseline_documents()
    print(f"  基线文档块总数: {len(base_docs)}")
    build_kb(docs=base_docs, persist_directory=config.BASE_DB_DIR, force_rebuild=args.rebuild)
    print("  基线向量库:", kb_stats(config.BASE_DB_DIR))

    print("=" * 68)
    print("【3/3】CLIP 图像向量库状态")
    print("=" * 68)
    print("  图像索引:", get_image_index().stats())

    print("-" * 68)
    print(f"知识库构建完成，用时 {time.time() - t0:.1f}s")
    print(f"  优化后: {config.DB_DIR}")
    print(f"  基线  : {config.BASE_DB_DIR}")
    print(f"  图像  : {config.IMAGE_DB_DIR}")


if __name__ == "__main__":
    main()