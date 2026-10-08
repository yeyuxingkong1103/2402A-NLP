# 工单编号：人工智能NLP-RAG-LightRAG优化
# src/lightrag_v12/__init__.py
from .lightrag_wrapper import get_lightrag, lightrag_query, lightrag_insert_text

__all__ = ["get_lightrag", "lightrag_query", "lightrag_insert_text"]
