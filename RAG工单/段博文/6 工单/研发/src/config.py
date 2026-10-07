# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
全局配置模块（混合检索策略增强版）：在工单05基础上，提供可配置的多种检索策略。

核心优化点：
    1. 检索策略：vector（向量召回+重排）/ fulltext（全文检索）/ hybrid（混合检索）三种模式可配置；
    2. 全文检索：倒排索引 + 布尔查询(AND/OR/NOT) + 短语匹配("...") + 模糊匹配(~) + 多字段加权；
    3. 多种重排：cross_encoder（bge-reranker）/ tfidf（TF-IDF重排器）/ llm（LLM重排器）；
    4. 多种融合：rrf（倒数排名融合）/ weighted（加权平均）/ voting（Borda投票），权重可调；
    5. 继承 05：Query改写/多轮对话；继承 04：图像语义解析；继承 03：表格解析。
"""

import os

# ==================== 一、Milvus 向量数据库 ====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# 优化：独立集合，用图像语义增强后的数据重新入库
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_pdf_qa_v4")
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

# ==================== 五-二、混合检索策略配置（工单06核心新增） ====================
# 检索策略：vector（向量检索：召回+重排）/ fulltext（全文检索）/ hybrid（混合检索）
SEARCH_STRATEGY = os.getenv("SEARCH_STRATEGY", "hybrid")
# 融合算法：rrf（倒数排名融合）/ weighted（加权平均）/ voting（Borda计数投票）
FUSION_ALGORITHM = os.getenv("FUSION_ALGORITHM", "rrf")
# 重排算法：cross_encoder（bge-reranker）/ tfidf（TF-IDF重排）/ llm（LLM打分重排）
RERANK_ALGORITHM = os.getenv("RERANK_ALGORITHM", "cross_encoder")
# 混合检索权重（weighted 融合时生效，自动归一化）
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))
FULLTEXT_WEIGHT = float(os.getenv("FULLTEXT_WEIGHT", "0.4"))
# RRF 平滑常数
RRF_K = int(os.getenv("RRF_K", "60"))
# 全文检索多字段权重（倒排索引：标题/正文/内容类型）
FIELD_WEIGHT_TITLE = float(os.getenv("FIELD_WEIGHT_TITLE", "3.0"))
FIELD_WEIGHT_CONTENT = float(os.getenv("FIELD_WEIGHT_CONTENT", "1.0"))
FIELD_WEIGHT_TYPE = float(os.getenv("FIELD_WEIGHT_TYPE", "2.0"))
# 全文检索开关：是否启用查询语法解析（布尔/短语/模糊）
FULLTEXT_QUERY_SYNTAX = os.getenv("FULLTEXT_QUERY_SYNTAX", "true").lower() == "true"
# 模糊匹配最大编辑距离
FUZZY_MAX_DISTANCE = int(os.getenv("FUZZY_MAX_DISTANCE", "1"))
# LLM 重排候选数上限（控制成本）
LLM_RERANK_TOP_N = int(os.getenv("LLM_RERANK_TOP_N", "10"))

# ==================== 六、表格解析配置（继承工单03） ====================
TABLE_EXTRACTION_ENABLED = os.getenv("TABLE_EXTRACTION_ENABLED", "true").lower() == "true"
TABLE_EXTRACTOR = os.getenv("TABLE_EXTRACTOR", "pdfplumber")
TABLE_TO_MARKDOWN = os.getenv("TABLE_TO_MARKDOWN", "true").lower() == "true"
TABLE_MIN_ROWS = int(os.getenv("TABLE_MIN_ROWS", "2"))

# ==================== 七、图像内容解析配置（核心新增） ====================
IMAGE_EXTRACTION_ENABLED = os.getenv("IMAGE_EXTRACTION_ENABLED", "true").lower() == "true"
# 图像渲染缩放倍数（越高 OCR 越清晰、越慢）
IMAGE_RENDER_ZOOM = float(os.getenv("IMAGE_RENDER_ZOOM", "2.0"))
# 矢量绘图"笔画数"阈值：超过则认为该页含结构图/统计图（矩形、连线较多）
IMAGE_STROKE_MIN = int(os.getenv("IMAGE_STROKE_MIN", "45"))
# 内嵌位图视为"真实图片"的最小宽与高（双维判定，排除 1x1304 的线条色块
# 和 143x127 的水印小图）
IMAGE_BITMAP_MIN_W = int(os.getenv("IMAGE_BITMAP_MIN_W", "200"))
IMAGE_BITMAP_MIN_H = int(os.getenv("IMAGE_BITMAP_MIN_H", "150"))
# 单个 PDF 最多解析的图像页数（控制 LLM 调用成本）
IMAGE_MAX_PER_PDF = int(os.getenv("IMAGE_MAX_PER_PDF", "40"))
# OCR 信息密度门槛：有效文字框数 / 总字数低于此值判为照片/印章/Logo，不送 LLM
IMAGE_MIN_OCR_BOXES = int(os.getenv("IMAGE_MIN_OCR_BOXES", "4"))
IMAGE_MIN_OCR_CHARS = int(os.getenv("IMAGE_MIN_OCR_CHARS", "20"))
# 不含百分比时（结构图/流程图等）要求的更高字数，用于剔除凭证/图章类扫描噪声
IMAGE_CHART_MIN_CHARS = int(os.getenv("IMAGE_CHART_MIN_CHARS", "150"))
# 图引用/图题触发词（用于判定该页含图）
IMAGE_CAPTION_KEYWORDS = os.getenv(
    "IMAGE_CAPTION_KEYWORDS",
    "如下图,见图,如图,下图所示,图示,结构如下图,见下图,如下图所示",
).split(",")
# OCR 结果中需要忽略的水印词
IMAGE_WATERMARKS = ["八维教育", "八维", "rengong", "人工智能刘", "智能刘", "zhineng"]

# ==================== 八、结果缓存 ====================
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))

# ==================== 九、FastAPI 服务 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ==================== 十、日志 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "rag_pdf_qa_v6.log")
