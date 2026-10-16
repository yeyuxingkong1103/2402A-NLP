# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单8 Graph RAG 包：知识图谱构建 + 图谱增强检索 + 问答。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

模块地图：
    * ``config``    —— 路径与**全部阈值**（环境变量可覆盖），索引直接引用工单7 绝对路径；
    * ``schema``    —— 实体类型表 / 关系类型表（唯一真源，pytest 校验）；
    * ``extractor`` —— 规则/词典/正则抽取器（+ 可选 Ollama 抽样补抽）；
    * ``store``     —— 图谱落盘/加载（entities.jsonl / relations.jsonl / graph.json / GraphML）；
    * ``llm``       —— Ollama ``qwen2.5:3b`` 生成与关系补抽客户端（带超时与降级）；
    * ``engine``    —— Graph RAG 检索与问答引擎（向量+BM25 召回 → 图谱一跳/两跳扩展 → 生成）；
    * ``visualize`` —— PIL 自绘力导向布局 PNG + 自包含 SVG/HTML。
"""

from __future__ import annotations

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

__all__ = ["WORK_ORDER"]
