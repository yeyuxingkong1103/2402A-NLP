# -*- coding: utf-8 -*-
"""定位 langchain_text_splitters 导入慢的子模块（工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化）"""
from __future__ import annotations

import importlib
import os
import time

# 先按 .env 的方式设置 HF 离线变量，验证「未设置离线 → 联网超时」假设
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SUBS = [
    "torch",
    "transformers",
    "sentence_transformers",
    "langchain_text_splitters.base",
    "langchain_text_splitters.sentence_transformers",
    "langchain_text_splitters.spacy",
    "langchain_text_splitters.konlpy",
    "langchain_text_splitters.nltk",
    "langchain_text_splitters",
]

for m in SUBS:
    t = time.time()
    try:
        importlib.import_module(m)
        print(f"{m:<52s} {time.time() - t:7.2f}s", flush=True)
    except Exception as exc:
        print(f"{m:<52s} {time.time() - t:7.2f}s  (跳过: {type(exc).__name__})", flush=True)