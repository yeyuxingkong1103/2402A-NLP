# -*- coding: utf-8 -*-
"""导入链路逐模块计时（工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化）"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODULES = [
    "langchain_core.documents",
    "langchain_ollama",
    "langchain_community.vectorstores",
    "langchain_core.prompts",
    "rank_bm25",
    "jieba",
    "fitz",
    "rapidocr_onnxruntime",
    "src.config",
    "src.pdf_parser",
    "src.table_parser",
    "src.chunking",
    "src.reranker",
    "src.retriever",
    "src.knowledge_base",
]

for m in MODULES:
    t = time.time()
    try:
        importlib.import_module(m)
        print(f"{m:<40s} {time.time() - t:7.2f}s", flush=True)
    except Exception as exc:
        print(f"{m:<40s} FAILED {exc}", flush=True)