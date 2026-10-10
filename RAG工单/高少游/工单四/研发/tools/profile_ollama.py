# -*- coding: utf-8 -*-
"""定位 langchain_ollama 导入慢的子模块（工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化）"""
from __future__ import annotations

import importlib
import time

SUBS = [
    "ollama",
    "pydantic",
    "langchain_ollama.embeddings",
    "langchain_ollama.llms",
    "langchain_ollama.chat_models",
    "langchain_ollama",
]

for m in SUBS:
    t = time.time()
    try:
        importlib.import_module(m)
        print(f"{m:<40s} {time.time() - t:7.2f}s", flush=True)
    except Exception as exc:
        print(f"{m:<40s} {time.time() - t:7.2f}s  (跳过: {type(exc).__name__}: {exc})", flush=True)