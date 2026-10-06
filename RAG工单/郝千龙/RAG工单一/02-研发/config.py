# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【全局配置 · config.py】集中管理 RAG 系统全部可调参数、路径与 Prompt 模板
# 编写日期：2026-09-28   修订日期：2026-10-04
import os
from pathlib import Path

# 项目根目录（研发代码所在目录的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据源 PDF（必须放在工单根目录）
SOURCE_PDF = BASE_DIR / "招股说明书1.pdf"

# 各类工作目录
DATA_DIR = BASE_DIR / "02-研发" / "data"          # 缓存与中间产物
MILVUS_DIR = DATA_DIR / "milvus"                 # Milvus 本地存储
CHROMA_DIR = DATA_DIR / "chroma"                 # Chroma 本地持久化目录
FAISS_DIR = DATA_DIR / "faiss"                   # FAISS 索引与元数据目录
JSONL_CACHE = DATA_DIR / "parsed.jsonl"          # PDF 解析结果缓存
UPLOAD_DIR = DATA_DIR / "uploads"                # 用户上传的 PDF
BM25_CACHE = DATA_DIR / "bm25_index.pkl"         # BM25 索引持久化文件
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"      # 用户反馈记录

for _d in (DATA_DIR, MILVUS_DIR, CHROMA_DIR, FAISS_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- PDF 解析 ----------
PDF_PARSER_BACKEND = "pymupdf"   # 可选: pymupdf / pdfplumber
PDF_TABLE_EXTRACT = True         # 是否提取表格

# ---------- 文本切分 ----------
CHUNK_SIZE = 512                 # 切片最大字符数
CHUNK_OVERLAP = 64               # 切片重叠字符数
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""]

# ---------- Embedding ----------
# bge-base-zh-v1.5：本地已完整缓存，768 维，官方支持中英文检索；CPU 速度快
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-base-zh-v1.5")
EMBED_DIM = 768                  # bge-base 向量维度
EMBED_BATCH_SIZE = 32
# 当前机器无 GPU，默认 CPU；有 CUDA 时设置环境变量 RAG_USE_GPU=1
EMBED_DEVICE = "cuda" if os.environ.get("RAG_USE_GPU", "0") == "1" else "cpu"

# ---------- 向量库后端 ----------
# faiss：工业级本地向量索引，单文件持久化，Windows 稳定免部署（默认）
# chroma：本地向量库（可选，个别版本 Windows HNSW 持久化存在兼容问题）
# milvus：生产级分布式向量库，需独立部署 Milvus 服务
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "faiss")

# ---------- Milvus 参数（VECTOR_BACKEND=milvus 时生效） ----------
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "rag_prospectus"
MILVUS_INDEX_TYPE = "HNSW"       # 优化方案 P0：IVF_FLAT → HNSW
MILVUS_METRIC_TYPE = "COSINE"
MILVUS_NLIST = 128               # IVF 聚类中心数（回退索引时使用）
MILVUS_HNSW_M = 16               # HNSW 每层出度
MILVUS_HNSW_EF = 64              # HNSW 查询时候选队列

# ---------- 缓存后端 ----------
# local：本地 JSON 文件缓存，无需 Redis（默认）
# redis：生产缓存，需独立部署 Redis
CACHE_BACKEND = os.environ.get("CACHE_BACKEND", "local")

# ---------- Redis 参数（CACHE_BACKEND=redis 时生效） ----------
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = 0
CACHE_TTL = 3600                 # 缓存过期秒数
ENABLE_CACHE = True

# ---------- LLM ----------
# 默认使用 DeepSeek 云端 API（OpenAI 兼容协议）
# 通过环境变量 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL 注入
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)
LLM_API_KEY = os.environ.get(
    "LLM_API_KEY",
    os.environ.get("DEEPSEEK_API_KEY", "sk-demo"),
)
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 400             # 控制生成长度，保障 <3s SLA
LLM_TIMEOUT = 8                  # 单次调用超时（秒）

# ---------- 检索参数 ----------
RETRIEVE_TOP_K = 5               # 最终返回片段数
RECALL_NUM = 20                  # 召回候选数（rerank/RRF 前）

# ---------- Rerank 重排（优化方案 P0） ----------
# GPU 环境默认开启；CPU 单机环境重排模型(2.2GB)加载慢，默认关闭，可用 ENABLE_RERANK=1 强制
_HAS_GPU = os.environ.get("RAG_USE_GPU", "0") == "1"
ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "1" if _HAS_GPU else "0") == "1"
RERANK_MODEL_NAME = "BAAI/bge-reranker-large"  # 本地已缓存
RERANK_BATCH_SIZE = 8
RERANK_TIMEOUT_MS = 800          # 重排超时保护，超时自动降级

# ---------- 多路召回（优化方案 P1） ----------
ENABLE_BM25 = True               # 关键词路径：公司名/年份/数字召回
RRF_K = 60                       # Reciprocal Rank Fusion 平滑常数
BM25_TOP_N = 20                  # BM25 路径召回数

# ---------- FastAPI ----------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# ---------- 工单 10 题验收清单 ----------
WORKORDER_QUESTIONS = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95,  "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33,  "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34,  "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]

# ---------- 中文 RAG Prompt ----------
PROMPT_TEMPLATE_ZH = """你是基于《招股说明书》的专业问答助手。
请严格仅依据以下检索到的文档片段回答问题，不得编造片段中没有的信息。

术语等价规则：本招股书中"军用/军工/军方/国防/国防客户/国防领域"均指同一军工国防领域；
若问题问"军用领域收入"，则片段中表述为"国防客户销售额""国防领域收入"的金额同样是答案，必须如实提取，不得回答"未提及"。

提取要求：
1. 涉及多期数据时，必须列出片段中出现的全部期间数值（如各年度/各期的金额或占比），不得遗漏；
2. 先给结论再列关键数据，准确简洁；只回答所问，不要添加"关键信息"等额外展开或片段未支持的细节；
3. 在回答末尾用"（依据：第X页）"标注引用页码；
4. 仅当片段及其同义表述中确实完全没有该信息时，才回答："招股说明书中未提及该信息。"

【检索片段】
{context}

【问题】
{question}

【回答】
"""

# ---------- 英文 RAG Prompt（多语言验收） ----------
PROMPT_TEMPLATE_EN = """You are a professional Q&A assistant grounded on a Chinese stock prospectus.
Answer the question strictly based on the retrieved passages below. Do not fabricate information.

Terminology rule: in this prospectus "military / military-industry / national-defense / defense customers / defense sector"
all refer to the SAME military/defense domain. If the question asks for "military-sector revenue", amounts in the passages
worded as "defense customer sales" or "national-defense revenue" ARE the answer — extract them faithfully and never say the information is absent.

Requirements:
1. When multiple periods are involved, list ALL period values present in the passages (every year/period amount or ratio), do not omit any;
2. Give the conclusion first, then key figures; be concise and answer ONLY what is asked, without extra "key information" sections or unsupported details;
3. Cite source pages at the end as "(Source: p.X)";
4. Only reply "The prospectus does not mention this information." when no passage, even under synonymous wording, contains it.

【Retrieved Passages】
{context}

【Question】
{question}

【Answer】
"""

# 基线 Prompt（无检索，纯 LLM）
BASELINE_PROMPT_TEMPLATE_ZH = """请直接回答以下问题。
【问题】{question}
"""
BASELINE_PROMPT_TEMPLATE_EN = """Please answer the following question directly.
【Question】{question}
"""


def get_rag_prompt(lang: str = "zh") -> str:
    """按语言返回 RAG Prompt 模板"""
    return PROMPT_TEMPLATE_EN if lang == "en" else PROMPT_TEMPLATE_ZH


def get_baseline_prompt(lang: str = "zh") -> str:
    """按语言返回基线 Prompt 模板"""
    return BASELINE_PROMPT_TEMPLATE_EN if lang == "en" else BASELINE_PROMPT_TEMPLATE_ZH


def jsonl_cache_for(pdf_path) -> Path:
    """按 PDF 返回解析缓存路径：源招股书用固定 parsed.jsonl，其他文件按文件名区分，避免互相覆盖"""
    pdf_path = Path(pdf_path)
    if pdf_path.resolve() == SOURCE_PDF.resolve():
        return JSONL_CACHE
    safe = "".join(c for c in pdf_path.stem if c not in '\\/:*?"<>|')
    return DATA_DIR / f"parsed_{safe}.jsonl"


def tables_cache_for(pdf_path) -> Path:
    """按 PDF 返回表格提取结果缓存路径（pdfplumber 全量表格提取很慢，独立 pickle 缓存）"""
    pdf_path = Path(pdf_path)
    safe = "".join(c for c in pdf_path.stem if c not in '\\/:*?"<>|')
    return DATA_DIR / f"tables_{safe}.pkl"

# ====================================================================
# 技术备注：
# 1. Transformer：bge-m3 基于 XLM-RoBERTa 多语言 Transformer，经对比学习训练，
#    支持中英跨语言检索（英文 Query 可命中中文文档块）。
# 2. RAG：配置覆盖 RAG 全链路——切分→Embedding→多路召回→重排→生成→缓存。
# 3. Fine-tuning：默认免微调；如需领域适配，可用 InfoNCE 在问答对上微调 Embedding。
# ====================================================================
