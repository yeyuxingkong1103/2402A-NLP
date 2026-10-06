# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
全局配置模块（表格解析增强版）：在工单一基础上，增加表格结构化解析。

核心优化点：
    1. 表格解析：新增 TABLE_EXTRACTION 配置，启用 pdfplumber 提取 PDF 表格；
    2. 表格转 Markdown：表格数据转为 Markdown 格式，保留行列结构语义；
    3. 独立集合：新建 rag_pdf_qa_v3，存储表格增强后的文档块；
    4. 保留工单二优化：缓存、LLM 参数、检索策略全部继承。
"""

import os

# ==================== 一、Milvus 向量数据库 ====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# 优化：独立集合，用表格解析后的数据重新入库
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_pdf_qa_v3")
MILVUS_INDEX_TYPE = "AUTOINDEX"
MILVUS_METRIC_TYPE = "COSINE"

# ==================== 二、向量模型（本地 bge-m3 权重） ====================
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", r"D:\Projects\Models\bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED_DIM = 1024

# ==================== 三、重排模型（本地权重目录） ====================
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    r"D:\Projects\Models\BAAI--bge-reranker-large\bge-reranker-large",
)
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")

# ==================== 四、大模型（DeepSeek API） ====================
LLM_API_KEY = os.getenv("deepseek_api_key1")
LLM_BASE_URL = os.getenv("deepseek_base_url")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))

# ==================== 五、文本切分与检索参数 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RECALL = 6
TOP_K_RERANK = 3
SCORE_THRESHOLD = 0.25

# ==================== 六、表格解析配置（核心新增） ====================
TABLE_EXTRACTION_ENABLED = os.getenv("TABLE_EXTRACTION_ENABLED", "true").lower() == "true"
# 表格解析方式：pdfplumber（精确）或 fitz（快速）
TABLE_EXTRACTOR = os.getenv("TABLE_EXTRACTOR", "pdfplumber")
# 表格转 Markdown 格式，保留行列结构
TABLE_TO_MARKDOWN = os.getenv("TABLE_TO_MARKDOWN", "true").lower() == "true"
# 表格块最小行数（少于此行数不视为表格）
TABLE_MIN_ROWS = int(os.getenv("TABLE_MIN_ROWS", "2"))

# ==================== 七、结果缓存 ====================
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))

# ==================== 八、FastAPI 服务 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ==================== 九、日志 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "rag_pdf_qa_v3.log")
