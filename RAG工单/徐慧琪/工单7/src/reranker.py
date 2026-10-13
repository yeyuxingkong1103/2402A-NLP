# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：精排（Rerank）—— 兼容层

本工单的精排实现已升级为**注册表模式**（多模型可切换 + 多算法基线），
统一入口在 `src/rerankers.py`：
  · 4 种算法：model（真实 Reranker 模型，生产默认）/ llm / tfidf / adaptive
  · 3 个可切换模型：bge-reranker-v2-m3、bge-reranker-v2-m3-ms、bge-reranker-base

本文件保留为**向后兼容 shim**：上游 01~05 的代码与测试通过
`reranker.rerank(...)` / `reranker.get_model()` / `reranker.info()` 调用精排，
这里原样转发到 rerankers.py，避免改动上游调用点。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

from typing import Sequence

from src import rerankers as _impl


def rerank(query: str, candidates: Sequence[dict],
           top_k: int | None = None) -> list[dict]:
    """对候选块精排，返回 top_k 条（附 rerank_score）。任何失败都不抛出。"""
    return _impl.rerank(query, candidates, top_k=top_k)


def get_model(name: str | None = None):
    """懒加载精排模型，返回 (tokenizer, model, device)。"""
    return _impl.get_model(name)


def is_available(name: str | None = None) -> bool:
    return _impl.is_available(name)


def unload(name: str | None = None) -> None:
    _impl.unload(name)


def info() -> dict:
    return _impl.info()
