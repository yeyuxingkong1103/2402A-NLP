# -*- coding: utf-8 -*-
"""
配置模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""
import os

# ============ PDF 文档路径 ============
# 默认使用与工单同目录的招股说明书 PDF
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.environ.get(
    "PDF_PATH",
    os.path.join(BASE_DIR, "工单1", "招股说明书1.pdf"),
)

# ============ LLM 服务配置 ============
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# ============ 检索参数 ============
CHUNK_SIZE = 500          # 文本块字符数
CHUNK_OVERLAP = 50         # 文本块重叠字符数
TOP_K = 5                  # 检索返回的文本块数量

# ============ 性能要求 ============
MAX_RESPONSE_TIME = 3.0    # 最大响应时间 (秒)

# ============ 缓存与索引文件 ============
CACHE_DIR = os.path.join(BASE_DIR, "研发", "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "tfidf_index.pkl")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")

os.makedirs(CACHE_DIR, exist_ok=True)
