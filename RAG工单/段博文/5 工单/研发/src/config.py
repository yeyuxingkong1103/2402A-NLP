# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
"""
全局配置模块（图像内容解析增强版）：在工单01/02/03 基础上，增加 PDF 图像语义解析。

核心优化点：
    1. 图像解析：新增 IMAGE 配置，自动定位 PDF 中的结构图/统计图（含矢量图）；
    2. 多模态识别：使用 RapidOCR（深度视觉模型）识别图中中文文字与数字；
    3. 语义生成：将 OCR 结果结合图题交给 LLM，生成结构化、可检索的图像语义块；
    4. 独立集合：新建 rag_pdf_qa_v4，文本/表格/图像三类块统一存储；
    5. 继承 03：表格 Markdown 解析；继承 02：缓存、LLM 参数、检索策略。
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
LOG_FILE = os.getenv("LOG_FILE", "rag_pdf_qa_v4.log")
