# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：优化前系统（工单1）单题问答桥接脚本 —— 独立子进程运行

为什么用子进程：
  工单1 与工单2 的包名同为 `src`，同一进程内无法共存（sys.modules 冲突）。
  子进程以工单1为工作目录与 sys.path 根，天然隔离，且**只读**调用
  工单1 的代码与索引，不修改其任何文件。

用法：
    python scripts/baseline_ask.py "公司注册资本是多少？"
输出：最后一行 JSON（工单1 RagResult.to_dict()）
"""
from __future__ import annotations

import json
import os
import sys

_B1 = os.environ.get("BASELINE_ROOT", r"D:\xinzg6\zy-gq\工单1")
sys.path.insert(0, _B1)
os.chdir(_B1)

from src import bootstrap  # noqa: E402


def _work() -> None:
    from src import rag

    # 单题：argv[1]；批量：--stdin（每行一题）
    if len(sys.argv) > 1 and sys.argv[1] != "--stdin":
        questions = [sys.argv[1]]
    else:
        questions = [line.strip() for line in sys.stdin if line.strip()]
    if not questions:
        questions = ["公司注册资本是多少？"]

    for question in questions:
        result = rag.ask(question)
        print("<<<RESULT>>>", flush=True)
        print(json.dumps(result.to_dict(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    bootstrap.run_with_large_stack(_work)
