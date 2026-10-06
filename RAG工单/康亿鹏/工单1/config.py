# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：全局配置。所有可调参数集中在本文件，便于统一维护。
          API Key 与 URL 从环境变量读取（参见 .env.example），其余常量直接在此定义。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 该编号会写入日志、界面与导出结果，便于交付验收追溯。
# ---------------------------------------------------------------------------
WORK_ORDER_NO = "人工智能NLP-RAG-基于PDF文档的问答系统"

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = BASE_DIR / "output"

for _d in (DATA_DIR, RAW_DIR, DOCS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# .env 中保存 API Key / 服务地址等敏感信息
load_dotenv(BASE_DIR / ".env")

# 默认解析的 PDF 文档（工单指定的《招股说明书1.pdf》）
DEFAULT_PDF_PATH = str(RAW_DIR / "招股说明书1.pdf")

# 工单给定的 10 条检索问题
QUESTIONS_FILE = str(DATA_DIR / "questions.json")

# 评估结果输出目录
EVAL_OUTPUT_DIR = OUTPUT_DIR

# ---------------------------------------------------------------------------
# 文档解析与切分配置
# ---------------------------------------------------------------------------
# 去除页眉页脚时跳过的上/下边距比例（0.05 表示上下各 5%）
PAGE_MARGIN_RATIO = 0.05
# 是否解析 PDF 中的表格（工单要求“文字、表格数据”均可解析）
ENABLE_TABLE_PARSING = True
# 文本切分窗口（字符数）与重叠
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
# 中文标点优先切分符
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " ", ""]

# ---------------------------------------------------------------------------
# 向量检索配置
# ---------------------------------------------------------------------------
# 向量库类型（本工单使用 milvus）
VECTOR_STORE_TYPE = "milvus"
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
# 集合名称
MILVUS_COLLECTION = "pdf_qa_prospectus"
MILVUS_DIM = 1024  # BGE-M3 dense 向量维度
MILVUS_METRIC = "IP"  # 内积（BGE-M3 向量已归一化，等价余弦相似度）

# 召回数量与重排后保留数量
RETRIEVE_TOP_K = 10
RERANK_TOP_N = 3
# 相似度下限，低于该值视为未命中（用于容错提示）
SCORE_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------
# 嵌入模型（BGE-M3）
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", EMBEDDING_MODEL_NAME)
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MAX_LENGTH = 1024
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "")  # 留空则自动选择 cuda/cpu
# BGE-M3 是否启用稀疏向量（本工单只用稠密检索，预留开关）
EMBEDDING_USE_SPARSE = False

# 重排模型（BGE-Reranker）
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", RERANKER_MODEL_NAME)
RERANKER_BATCH_SIZE = 8
RERANKER_MAX_LENGTH = 512
RERANKER_USE_FP16 = True

# 大语言模型（DeepSeek，OpenAI 兼容接口）
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Query 理解配置
# ---------------------------------------------------------------------------
QUERY_UNDERSTANDING_ENABLED = True
# 复杂问题分解后允许的最大子问题数量
MAX_SUB_QUESTIONS = 4

# ---------------------------------------------------------------------------
# 交互与运行配置
# ---------------------------------------------------------------------------
# 响应时间目标（秒），工单要求从提问到返回答案不超过 3 秒
RESPONSE_TIME_TARGET = 3.0
# 多语言支持：中文 / 英文
SUPPORTED_LANGUAGES = ["zh", "en"]
# 日志级别
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 国内访问 HuggingFace 的镜像（仅当环境变量未设置时生效）
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
