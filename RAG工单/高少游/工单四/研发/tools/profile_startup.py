# -*- coding: utf-8 -*-
"""启动耗时剖析（工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化）

逐段计时 app.py 启动路径上的关键步骤，定位「首屏加载慢」的瓶颈。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def step(label: str):
    t = time.time()

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            print(f"  {label:<46s} {time.time() - t:7.2f}s", flush=True)

    return _Ctx()


print("== 启动耗时剖析 ==", flush=True)
with step("import src.config"):
    from src import config
with step("import src.knowledge_base (含 langchain/faiss)"):
    from src.knowledge_base import kb_stats, get_all_documents
with step("import src.image_index"):
    from src.image_index import get_image_index
with step("kb_stats(config.DB_DIR)"):
    print("    ->", kb_stats(config.DB_DIR), flush=True)
with step("get_all_documents(config.DB_DIR)"):
    print("    ->", len(get_all_documents(config.DB_DIR)), "docs", flush=True)
with step("get_image_index().stats()"):
    print("    ->", get_image_index().stats(), flush=True)
print("== 完成 ==", flush=True)