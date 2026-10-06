# -*- coding: utf-8 -*-
"""
配置模块 V9 - Graph RAG 优化
工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LLM 配置
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# 图谱配置
KG_FILE = os.path.join(BASE_DIR, "financial_kg.json")

# V9 优化参数
MAX_HOPS = 2                    # 最大遍历深度
MIN_EDGE_WEIGHT = 0.3           # 边权重阈值 (剪枝)
ENTITY_LINKING_TOP_K = 5        # 实体链接 Top-K
PATH_SCORE_ALPHA = 0.5          # 路径质量权重
TEXT_SCORE_BETA = 0.3           # 文本分数权重
LLM_RERANK_GAMMA = 0.2          # LLM 重排权重

# 评估
TARGET_CONTEXT_PRECISION = 0.80
TARGET_CONTEXT_RECALL = 0.90

# 缓存
CACHE_DIR = os.path.join(BASE_DIR, "cache_v9")
os.makedirs(CACHE_DIR, exist_ok=True)

PORT = 5008
