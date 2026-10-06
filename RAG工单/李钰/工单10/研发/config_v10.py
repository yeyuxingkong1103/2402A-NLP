# -*- coding: utf-8 -*-
"""
部署配置 - V10
工单编号: 人工智能 NLP-RAG-金融问答系统部署
"""
import os

# Flask
PORT = int(os.environ.get("FLASK_PORT", "5008"))
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")

# 数据目录 (Docker 卷挂载)
DATA_DIR = os.environ.get("RAG_DATA_DIR", "/app/data")
CACHE_DIR = os.environ.get("RAG_CACHE_DIR", "/app/cache")
LOG_DIR = os.environ.get("RAG_LOG_DIR", "/app/logs")
SHARED_DIR = os.environ.get("RAG_SHARED_DIR", "/app/shared")

# LLM
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# Neo4j (可选)
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "")

# 自动创建目录
for d in [DATA_DIR, CACHE_DIR, LOG_DIR, SHARED_DIR]:
    os.makedirs(d, exist_ok=True)
