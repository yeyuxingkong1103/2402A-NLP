# -*- coding: utf-8 -*-
"""legal_rag —— 法律 RAG 应用包。

分层：
  ingest/     文档解析、分块、入库编排（离线链路）
  embedding/  向量化（可插拔：BGE-m3 / 零重依赖兜底）
  store/      向量库（可插拔：ChromaDB / 内存）
  retrieve/   混合检索与重排（在线链路）
  generate/   提示词、大模型客户端、模型路由、后处理
  memory/     短期/长期记忆与会话隔离
  api/        FastAPI HTTP 接口
"""

__version__ = "0.1.0"
