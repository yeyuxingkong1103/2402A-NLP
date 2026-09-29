# app/config/settings.py
"""全局配置：统一从 .env 读取，已存在的环境变量优先。"""
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

# 项目根目录：config -> app -> 根
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# override=False：部署时以真实环境变量为准，.env 只作兜底
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)


def _env(key, default=None):
    v = os.getenv(key)
    return v if v not in (None, "") else default


def _int(key, default):
    try:
        return int(_env(key, default))
    except (TypeError, ValueError):
        return default


def _float(key, default):
    try:
        return float(_env(key, default))
    except (TypeError, ValueError):
        return default


# ==================== 大模型 ====================
LLM_API_KEY = _env("DEEPSEEK_API_KEY")
LLM_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = _env("DEEPSEEK_MODEL", "deepseek-flash")
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 4096)
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.7)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 120)

# ==================== Milvus ====================
MILVUS_URI = _env("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = _env("MILVUS_TOKEN", "")
MILVUS_COLLECTION = _env("MILVUS_COLLECTION", "rag_knowledge")
MILVUS_MEMORY_COLLECTION = _env("MILVUS_MEMORY_COLLECTION", "long_term_memory")

# ==================== Redis ====================
REDIS_HOST = _env("REDIS_HOST", "localhost")
REDIS_PORT = _int("REDIS_PORT", 6379)
REDIS_PASSWORD = _env("REDIS_PASSWORD")
REDIS_DB = _int("REDIS_DB", 0)

# ==================== MySQL ====================
MYSQL_HOST = _env("MYSQL_HOST", "localhost")
MYSQL_PORT = _int("MYSQL_PORT", 3306)
MYSQL_USER = _env("MYSQL_USER", "root")
MYSQL_PASSWORD = _env("MYSQL_PASSWORD", "")
MYSQL_DATABASE = _env("MYSQL_DATABASE", "rag_roleplay")
MYSQL_CHARSET = _env("MYSQL_CHARSET", "utf8mb4")


def mysql_url(with_database: bool = True) -> str:
    """SQLAlchemy 连接串。密码可能含 @ 等字符，必须转义。"""
    pwd = quote_plus(MYSQL_PASSWORD) if MYSQL_PASSWORD else ""
    auth = "%s:%s@" % (MYSQL_USER, pwd) if pwd else "%s@" % MYSQL_USER
    return "mysql+mysqlconnector://%s%s:%d/%s?charset=%s" % (
        auth, MYSQL_HOST, MYSQL_PORT,
        MYSQL_DATABASE if with_database else "", MYSQL_CHARSET)


# ==================== 本地模型 ====================
EMBEDDING_MODEL_PATH = _env("EMBEDDING_MODEL_PATH", "/root/models/bge-m3")
RERANKER_MODEL_PATH = _env("RERANKER_MODEL_PATH", "/root/models/bge-reranker-v2-m3")
EMBEDDING_DIM = _int("EMBEDDING_DIM", 1024)

# ==================== PDF 解析（MinerU） ====================
# MinerU 装在项目内的独立 venv 里（依赖与主环境冲突，见 mineru_service 模块注释）。
# 找不到这个可执行文件时，pdf_service 会退回 PyMuPDF 纯文本层提取。
MINERU_BIN = _env("MINERU_BIN", os.path.join(BASE_DIR, "mineru/.venv/bin/mineru-kit"))
# 解析档位。basic = 小模型（版面+OCR+公式+表格），无需 VLM，跑到 GPU 上。
# standard/advanced 要多跑一个 VLM，而 VLM 引擎只有 CPU 版 llama.cpp，
# 单次加载约 361 秒，子进程模式下每次调用都要重付，故默认 basic。
MINERU_TIER = _env("MINERU_TIER", "basic")
# .env 里可能写成相对路径（如 ./mineru/.venv/bin/mineru-kit）。
# 相对路径要相对项目根解析 —— 否则换工作目录启动就找不到可执行文件。
if MINERU_BIN and not os.path.isabs(MINERU_BIN):
    MINERU_BIN = os.path.normpath(os.path.join(BASE_DIR, MINERU_BIN))
# 模型源。huggingface.co 在本机直连不通，必须走 modelscope。
MINERU_MODEL_SOURCE = _env("MINERU_MODEL_SOURCE", "modelscope")
# 单份文档的解析超时（秒）。basic 档跑 GPU，实测 28 页约 26 秒，留足余量。
MINERU_TIMEOUT = _int("MINERU_TIMEOUT", 1800)

# ==================== 记忆策略 ====================
SHORT_TERM_TURNS = _int("SHORT_TERM_TURNS", 10)
LONG_TERM_TOP_K = _int("LONG_TERM_TOP_K", 3)
MEMORY_SUMMARY_TRIGGER = _int("MEMORY_SUMMARY_TRIGGER", 6)

# ==================== 检索策略 ====================
RETRIEVE_LIMIT = _int("RETRIEVE_LIMIT", 20)
RERANK_TOP_K = _int("RERANK_TOP_K", 5)
# 查询扩写条数，1 表示关闭扩写
QUERY_EXPAND_N = _int("QUERY_EXPAND_N", 3)
# 是否启用元数据路召回
ENABLE_METADATA_ROUTE = _env("ENABLE_METADATA_ROUTE", "true").lower() in ("1", "true", "yes")
# 是否启用生成结果校验（法条引用防幻觉）
ENABLE_CITATION_CHECK = _env("ENABLE_CITATION_CHECK", "true").lower() in ("1", "true", "yes")

# ==================== 目录 ====================
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

for _d in (UPLOADS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# 支持的角色（与 data/ 下的目录名一致）
SUPPORTED_ROLES = ["lawyer", "psychologist", "financial_advisor"]
