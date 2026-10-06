# -*- coding: utf-8 -*-
"""
配置模块 (优化版 V2)
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
"""
import os

# ============ 根目录 ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ PDF 文档路径 (工单2使用独立的招股说明书) ============
PDF_PATH = os.environ.get(
    "PDF_PATH",
    os.path.join(BASE_DIR, "招股说明书1.pdf"),
)

# ============ LLM 服务配置 ============
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# ============ 语义分块参数 (V2 优化) ============
MIN_PARA_LEN = 100          # 段落最小长度 (短于该值会合并)
MAX_PARA_LEN = 800          # 段落最大长度 (长于该值会拆分)
MERGE_OVERLAP = 80          # 合并时的重叠字符数

# ============ 混合检索参数 (V2 优化) ============
TOP_K_RAW = 20              # 检索返回的原始候选数 (Rerank 前)
TOP_K_FINAL = 5             # Rerank 后的最终返回数
BM25_WEIGHT = 0.5           # BM25 在混合检索中的权重
TFIDF_WEIGHT = 0.5          # TF-IDF 在混合检索中的权重

# ============ 性能要求 ============
MAX_RESPONSE_TIME = 3.0

# ============ 缓存与索引 ============
CACHE_DIR = os.path.join(BASE_DIR, "cache_v2")
INDEX_PATH = os.path.join(CACHE_DIR, "hybrid_index.pkl")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks_v2.json")

os.makedirs(CACHE_DIR, exist_ok=True)
