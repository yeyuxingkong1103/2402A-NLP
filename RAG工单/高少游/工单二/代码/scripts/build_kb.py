# -*- coding: utf-8 -*-
"""知识库构建脚本（优化版）
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

用法（项目根目录，激活 langchain2 环境，需已启动 Ollama）：
    python scripts/build_kb.py                # 构建【优化后】向量库
    python scripts/build_kb.py --baseline     # 构建【优化前】基线向量库
    python scripts/build_kb.py --force        # 强制重建
    python scripts/build_kb.py --all          # 同时构建优化前/后
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.knowledge_base import build_kb, build_baseline_documents, build_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAG 向量知识库")
    parser.add_argument("--baseline", action="store_true", help="构建基线（优化前）向量库")
    parser.add_argument("--all", action="store_true", help="同时构建优化前/后向量库")
    parser.add_argument("--force", action="store_true", help="强制重建")
    args = parser.parse_args()

    jobs = []
    if args.all:
        jobs = [(True, config.BASE_DB_DIR), (False, config.DB_DIR)]
    elif args.baseline:
        jobs = [(True, config.BASE_DB_DIR)]
    else:
        jobs = [(False, config.DB_DIR)]

    for baseline, path in jobs:
        tag = "优化前(基线)" if baseline else "优化后"
        docs = build_baseline_documents() if baseline else build_documents()
        print(f"[{tag}] 分块完成：{len(docs)} 块（表格块 "
              f"{sum(1 for d in docs if d.metadata.get('ctype') == 'table')}）")
        t0 = time.time()
        build_kb(persist_directory=path, force_rebuild=args.force, baseline=baseline)
        print(f"[{tag}] 向量库就绪：{path}  用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()