# -*- coding: utf-8 -*-
"""
配置模块 V6 - 检索策略配置化
工单编号: 人工智能 NLP-RAG-混合检索任务
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 检索策略配置 ============
# strategy: vector / fulltext / hybrid
RETRIEVAL_STRATEGY = os.environ.get("RETRIEVAL_STRATEGY", "hybrid")

# 混合检索权重
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", "0.6"))
FULLTEXT_WEIGHT = float(os.environ.get("FULLTEXT_WEIGHT", "0.4"))

# 嵌入模型: tfidf / bge / m3e / sentence-transformers
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "tfidf")

# 重排器: llm / tfidf / feedback / none
RERANKER = os.environ.get("RERANKER", "tfidf")

# 融合方法: weighted / rrf / vote
FUSION_METHOD = os.environ.get("FUSION_METHOD", "weighted")

# 召回/重排数量
TOP_K_RECALL = int(os.environ.get("TOP_K_RECALL", "20"))
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", "5"))

# ============ LLM 配置 (复用) ============
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# ============ 缓存 ============
CACHE_DIR = os.path.join(BASE_DIR, "cache_v6")
os.makedirs(CACHE_DIR, exist_ok=True)

# ============ 反馈存储 ============
FEEDBACK_FILE = os.path.join(CACHE_DIR, "user_feedback.json")

# 端口
PORT = 5005
