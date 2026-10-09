# -*- coding: utf-8 -*-
"""
LightRAG 配置
工单编号: 人工智能 NLP-RAG 项目-LightRAG 优化
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LLM
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# LightRAG 核心参数
MAX_TOKEN_SIZE = 800           # 文本块大小
OVERLAP = 100                  # 切块重叠
ENTITY_SUMMARY_MAX_LENGTH = 500 # 实体摘要长度
RELATION_SUMMARY_MAX_LENGTH = 200 # 关系摘要长度

# 双层检索
LOCAL_TOP_K = 5                 # 局部检索 Top-K
GLOBAL_TOP_K = 5                # 全局检索 Top-K
LOCAL_MAX_DEPTH = 2             # 局部子图深度
GLOBAL_MAX_DEPTH = 3            # 全局遍历深度
COMMUNITY_TOP_K = 3             # 社区数量

# 关键词分类阈值
LOCAL_KEYWORD_MIN_LEN = 2       # 局部关键词最小长度

# 路径
KG_FILE = os.path.join(BASE_DIR, "lightrag_kg.json")
INDEX_CACHE = os.path.join(BASE_DIR, "lightrag_index.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

# 端口
PORT = 5009
