# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/retrieval —— 工单六 混合检索策略包

- retrieval_config.py：检索策略配置（模式/融合/权重/重排器）
- fulltext_retriever.py：全文检索（倒排索引/布尔/短语/模糊/多字段 TF-IDF）
- rerankers.py：三种重排器（LLM 交叉编码 / TF-IDF / 用户反馈自适应）
- fusion.py：融合算法（RRF 投票 / 加权平均）
- hybrid_retriever_v6.py：统一可配置混合检索器
"""
