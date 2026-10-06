# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
全局配置模块（优化版）：在工单一基础上，针对响应时间、检索精度做参数调优。

核心优化点：
    1. 分块更细：CHUNK_SIZE 500→400，OVERLAP 80→50，提升向量召回精度；
    2. 召回收敛：TOP_K_RECALL 10→6，重排候选从 20 降到 12，CPU 重排耗时减半；
    3. 精排保留：TOP_K_RERANK 5→3，喂给 LLM 的上下文更聚焦，生成更快；
    4. 阈值下调：SCORE_THRESHOLD 0.3→0.25，配合更少候选量保住召回率；
    5. LLM 提速：MAX_TOKENS 2048→768、TEMPERATURE 0.3→0.1，生成更快更稳；
    6. 结果缓存：新增 CACHE 配置，重复查询命中缓存后毫秒级返回；
    7. 独立集合：新建 rag_pdf_qa_v2，与旧集合并存，便于前后对比。
"""

import os

# ==================== 一、Milvus 向量数据库 ====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# 优化：独立集合，用优化后的分块重新入库，旧集合保留作对比基线
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_pdf_qa_v2")
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
# 优化：温度降到 0.1，问答更稳定；max_tokens 砍到 768，事实型回答足够且更快
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))

# ==================== 五、文本切分与检索参数（核心调优区） ====================
# 优化：块从 500 缩到 400，语义更聚焦；重叠 80→50，减少冗余仍兜得住切口
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
# 优化：每路召回 10→6，重排候选减半，CPU 重排耗时大幅下降
TOP_K_RECALL = 6
# 优化：重排保留 5→3，喂给 LLM 的上下文更少更精，生成更快
TOP_K_RERANK = 3
# 优化：阈值 0.3→0.25，候选少了适当放低门槛保召回
SCORE_THRESHOLD = 0.25

# ==================== 六、结果缓存（新增） ====================
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))  # 缓存条数上限

# ==================== 七、FastAPI 服务 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ==================== 八、日志 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "rag_pdf_qa_v2.log")
