# -*- coding: utf-8 -*-
"""
配置模块 V8 - Graph RAG
工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LLM 配置 (复用)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# 图谱配置
KG_FILE = os.path.join(BASE_DIR, "financial_kg.json")
USE_NETWORKX = True  # NetworkX 图库 (内置轻量实现)
NEO4J_URI = os.environ.get("NEO4J_URI", "")  # 可选 Neo4j

# Graph RAG 检索参数
GRAPH_TOP_K_NODES = 10      # 初始实体链接 Top-K
GRAPH_NEIGHBOR_HOPS = 2     # 子图遍历深度
GRAPH_WEIGHT = 0.5          # 图谱分数权重
TEXT_WEIGHT = 0.3           # 文本分数权重

# 实体类型
ENTITY_TYPES = ["Company", "Person", "Product", "Standard", "Financial", "Industry"]

# 缓存
CACHE_DIR = os.path.join(BASE_DIR, "cache_v8")
os.makedirs(CACHE_DIR, exist_ok=True)

PORT = 5007
