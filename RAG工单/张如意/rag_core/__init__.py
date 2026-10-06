# -*- coding: utf-8 -*-
"""
rag_core —— RAG 项目共享核心库
工单编号：人工智能NLP-RAG（01~13 全部工单共用）

模块清单：
    config            全局配置（路径、模型、超参、评测问题集）
    llm               LLM 客户端（DeepSeek 主 / Ollama 备，带缓存与用量统计）
    embed             向量嵌入（BGE 系列，支持多模型切换）
    pdf_parse         PDF 解析（文字 / 表格 / 图像三层）
    chunk             分块（fixed / recursive / semantic / structure 四种策略）
    vectorstore       向量库（Chroma 主 / NumPy 备）
    bm25              全文检索（倒排索引 + BM25 + 布尔/短语/模糊查询）
    rerank            重排（LLM / TF-IDF / 自适应 三种 + 级联）
    retriever         统一检索（向量 / 全文 / 混合，三种融合算法）
    query_understand  Query 理解（意图 / 消歧 / 分解 / 多轮指代消解）
    generator         答案生成（RAG / 纯 LLM 基线 / 多跳）
    evaluate          RAG 评估（RAGAS 四大指标 + 检索层指标）
    image_parse       图像语义解析（CLIP / 多模态大模型）
    graph_rag         Graph RAG（知识图谱构建、社区检测、图谱检索）
    pipeline          流水线编排（各工单通过 PRESETS 切换能力）
    api               FastAPI 服务 + 内置 Web 问答界面
"""
from . import (bm25, chunk, config, embed, evaluate, generator,  # noqa: F401
               graph_rag, image_parse, llm, pdf_parse, pipeline,
               query_understand, rerank, retriever, vectorstore)

__version__ = "1.0.0"


def __getattr__(name: str):
    """
    惰性导出 `api` 子模块。

    `rag_core.api` 依赖 fastapi/uvicorn，属于按需使用的重依赖，
    因此不在包导入时立刻加载；但 `from rag_core import api` 仍可正常工作
    （由本函数在首次访问时才真正导入）。
    """
    if name == "api":
        import importlib
        mod = importlib.import_module("rag_core.api")
        globals()["api"] = mod
        return mod
    raise AttributeError(f"module 'rag_core' has no attribute {name!r}")

__all__ = [
    "config", "llm", "embed", "pdf_parse", "chunk", "vectorstore",
    "bm25", "rerank", "retriever", "query_understand", "generator",
    "evaluate", "image_parse", "graph_rag", "pipeline",
]
