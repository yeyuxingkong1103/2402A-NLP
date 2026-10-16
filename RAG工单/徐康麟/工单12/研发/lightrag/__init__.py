# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""LightRAG 子系统包（图谱构建 / 双层检索 / 融合 / 生成）。

模块：
    config          配置中心（阈值全配置化）
    ollama_client   Ollama 嵌入与生成
    corpus          PDF → 文本/表格/图像块
    vector_index    向量索引（FAISS）+ BM25
    entity_schema   招股说明书实体/关系 Schema 与规范化
    graph_extract   LLM 抽样抽取实体关系
    graph_store     图谱存储 / 别名归并 / 关系去重
    graph_viz       图谱可视化 PNG
    routing         局部/全局关键词分流
    double_retrieval 双层检索 + RRF 融合
    qa              答案生成（统一后端接口）
    logging_util    日志与落盘工具
"""

from __future__ import annotations

__all__ = [
    "config",
    "ollama_client",
    "corpus",
    "vector_index",
    "entity_schema",
    "graph_extract",
    "graph_store",
    "graph_viz",
    "routing",
    "double_retrieval",
    "qa",
    "logging_util",
]

__version__ = "1.0.0"
