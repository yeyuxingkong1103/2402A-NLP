# -*- coding: utf-8 -*-
"""构建向量知识库脚本
用法（在项目根目录，激活 langchain2 环境）：
    python scripts/build_kb.py [--rebuild]
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_base import build_kb, kb_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建知识库")
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    args = parser.parse_args()

    store = build_kb(force_rebuild=args.rebuild)
    print(f"知识库构建完成。统计: {kb_stats()}")


if __name__ == "__main__":
    main()