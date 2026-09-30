"""
config.py — 全局配置中心

本模块集中管理所有可调参数，不含任何业务逻辑。
优先级：项目根目录的 .env 文件 > 本文件中的默认值。

本地开发（LOCAL_MODE=True）不需要 Milvus / Redis，直接跑得起来；
上传到算力云后把 LOCAL_MODE 改成 False，即切换为生产实现。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- 基础路径

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
LEGAL_DIR = DATA_DIR / "legal"
MEDICAL_DIR = DATA_DIR / "medical"
ENGLISH_DIR = DATA_DIR / "english"
GENERATED_PDF_DIR = DATA_DIR / "generated_pdfs"

LOG_DIR = BASE_DIR / "logs"
STORAGE_DIR = BASE_DIR / "storage"  # LOCAL_MODE 下的本地持久化目录


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------- 运行模式

# True  = 本地轻量模式：内存向量库 + JSON 会话历史，无需 Milvus/Redis
# False = 生产模式：Milvus 向量库 + Redis 会话历史
LOCAL_MODE = _env_bool("LOCAL_MODE", True)

# 向量化后端，"hash" 为离线降级实现，生产环境请使用 "bge-m3"
EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "hash" if LOCAL_MODE else "bge-m3")

# PDF 解析是否启用重型组件（minerU / PaddleOCR），缺失时会自动降级
USE_MINERU = _env_bool("USE_MINERU", not LOCAL_MODE)
USE_OCR = _env_bool("USE_OCR", not LOCAL_MODE)

# ---------------------------------------------------------------- LLM 配置

# 当前生效的大模型服务商："deepseek"（测试阶段） 或 "qwen"（算力云 SGLang 部署）
LLM_PROVIDER = _env("LLM_PROVIDER", "deepseek")

# ===================== DeepSeek —— 测试阶段使用 =====================
# ↓↓↓ 请在这里填入你的 DeepSeek API Key ↓↓↓
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY", "")
# ↑↑↑ 留空即可，填写后即可调用 ↑↑↑
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ===================== Qwen —— 算力云 SGLang 部署 =====================
# 算力云上的地址形如 http://<公网IP>:8000/v1，部署好后把地址填到这里
QWEN_BASE_URL = _env("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = _env("QWEN_MODEL", "qwen27b")
# SGLang / vLLM 默认不校验密钥，若网关开了鉴权再填
QWEN_API_KEY = _env("QWEN_API_KEY", "EMPTY")

# LLM 通用生成参数
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.7)
# 生成阶段保留思维链，推理模型的思考算在同一个额度里；实测部分问题思考可长达 3000 token，
# 上限设小了会让正文为空或被截断（finish_reason=length），故留足余量。
# 该值是上限而非目标，只按实际用量计费。
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 8192)
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 120)

# 意图识别、查询改写、探活这类短任务关闭思维链：推理模型的思考会先占满 max_tokens，
# 导致正文返回为空。留空表示不传该参数（SGLang/vLLM 自部署的 Qwen 通常不认这个参数）。
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT", "none" if LLM_PROVIDER == "deepseek" else "")


def get_llm_config() -> dict:
    """返回当前生效的 LLM 连接配置，供 llm_client.py 使用。"""
    if LLM_PROVIDER == "qwen":
        return {
            "provider": "qwen",
            "base_url": QWEN_BASE_URL,
            "api_key": QWEN_API_KEY,
            "model": QWEN_MODEL,
        }
    return {
        "provider": "deepseek",
        "base_url": DEEPSEEK_BASE_URL,
        "api_key": DEEPSEEK_API_KEY,
        "model": DEEPSEEK_MODEL,
    }


# ---------------------------------------------------------------- Redis

REDIS_HOST = _env("REDIS_HOST", "127.0.0.1")
REDIS_PORT = _env_int("REDIS_PORT", 6379)
REDIS_DB = _env_int("REDIS_DB", 0)
REDIS_PASSWORD = _env("REDIS_PASSWORD", "")
# 短期记忆保留最近多少轮对话（一轮 = 用户提问 + 模型回答）
SHORT_TERM_ROUNDS = _env_int("SHORT_TERM_ROUNDS", 5)

# ---------------------------------------------------------------- MySQL

MYSQL_HOST = _env("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = _env_int("MYSQL_PORT", 3306)
MYSQL_USER = _env("MYSQL_USER", "root")
MYSQL_PASSWORD = _env("MYSQL_PASSWORD", "your_password")
MYSQL_DB = _env("MYSQL_DB", "rag_chat")

# ---------------------------------------------------------------- Milvus

MILVUS_URI = _env("MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_TOKEN = _env("MILVUS_TOKEN", "")

# 领域 -> 知识库集合名映射
KB_COLLECTIONS = {
    "legal": "kb_legal",
    "medical": "kb_medical",
    "english": "kb_english",
}
LONG_TERM_COLLECTION = "long_term_memory"

# ---------------------------------------------------------------- 向量模型

EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# 稠密向量维度，Milvus schema 与本地向量库必须保持一致。
# 随模型走：m3e-base 是 768，bge-m3 是 1024；换模型必须同时重建集合。
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 768 if "m3e" in EMBEDDING_MODEL else 1024)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)

# HuggingFace 下载源。huggingface.co 在国内常常直连不上（实测超时），走镜像才能拉权重。
# 必须在 import huggingface_hub 之前写进环境变量，所以在这里提前 set。
HF_ENDPOINT = _env("HF_ENDPOINT", "https://hf-mirror.com")
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

# ---------------------------------------------------------------- 检索参数

# 两路召回各取多少条，合并后再做重排
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 10)
BM25_TOP_K = _env_int("BM25_TOP_K", 10)
# 重排后最终保留给大模型的文档数
RERANK_TOP_K = _env_int("RERANK_TOP_K", 5)
# 知识库检索最终交给大模型的条数
KB_TOP_K = _env_int("KB_TOP_K", 5)
# 余弦相似度低于该阈值的检索结果直接丢弃
SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.3)
# 入库去重：相似度高于该值视为重复内容
DUPLICATE_THRESHOLD = _env_float("DUPLICATE_THRESHOLD", 0.95)
# BM25 原始分数归一化后的权重，与向量分数加权融合
HYBRID_VECTOR_WEIGHT = _env_float("HYBRID_VECTOR_WEIGHT", 0.5)
HYBRID_BM25_WEIGHT = _env_float("HYBRID_BM25_WEIGHT", 0.5)

# ---------------------------------------------------------------- 长期记忆

# 每轮对话召回多少条历史记忆融合进提示词
LONG_TERM_TOP_K = _env_int("LONG_TERM_TOP_K", 3)

# 召回阈值按向量后端区分。bge-m3 是真实语义向量，用需求规定的 0.7；
# 而 hash 是特征哈希，分数里不含语义信息（实测同一句原样重复也只到 0.688，
# 换个说法的追问只有 0.23~0.39），对 hash 套 0.7 会让长期记忆永远召不回来。
LONG_TERM_THRESHOLD_BGE = _env_float("LONG_TERM_THRESHOLD_BGE", 0.7)
LONG_TERM_THRESHOLD_HASH = _env_float("LONG_TERM_THRESHOLD_HASH", 0.3)

# 时间衰减系数（天）。越久远的记忆排序越靠后，但不会因此被阈值丢掉。
MEMORY_DECAY_TAU_DAYS = _env_float("MEMORY_DECAY_TAU_DAYS", 30.0)

# 每多少轮对话提取一次用户偏好与已确认事实（关系记忆）
PROFILE_EXTRACT_EVERY = _env_int("PROFILE_EXTRACT_EVERY", 5)

# ---------------------------------------------------------------- 分块参数

CHUNK_SIZE = _env_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 50)
# 父子块策略中，父块与子块的倍数关系
PARENT_CHILD_RATIO = _env_int("PARENT_CHILD_RATIO", 3)
# 质量过滤：短于该长度的块直接丢弃
MIN_CHUNK_LENGTH = _env_int("MIN_CHUNK_LENGTH", 10)
# PDF 解析产生的最小有效文本长度
MIN_PAGE_TEXT_LENGTH = _env_int("MIN_PAGE_TEXT_LENGTH", 10)

# ---------------------------------------------------------------- 日志

LOG_LEVEL = _env("LOG_LEVEL", "INFO")
# 按天切分，例如 logs/app_20260917.log
LOG_FILE_PATTERN = str(LOG_DIR / "app_{time:YYYYMMDD}.log")
LOG_RETENTION = _env("LOG_RETENTION", "30 days")
# 日志是否同时输出到控制台
LOG_TO_CONSOLE = _env_bool("LOG_TO_CONSOLE", True)


def long_term_threshold() -> float:
    """长期记忆的召回阈值，按当前向量后端选择（原因见上方两个阈值的注释）。"""
    return LONG_TERM_THRESHOLD_HASH if EMBEDDING_BACKEND == "hash" else LONG_TERM_THRESHOLD_BGE


def ensure_dirs() -> None:
    """确保运行期需要的目录都存在。"""
    for path in (LOG_DIR, STORAGE_DIR, LEGAL_DIR, MEDICAL_DIR, ENGLISH_DIR, GENERATED_PDF_DIR):
        path.mkdir(parents=True, exist_ok=True)


_logger_ready = False


def setup_logger():
    """初始化全局日志：控制台 + logs/app_YYYYMMDD.log 按天切分。

    各模块直接用 `from loguru import logger` 即可，本函数负责配置 sink。
    重复调用是安全的（不会重复添加 sink）。
    """
    global _logger_ready
    from loguru import logger

    if _logger_ready:
        return logger

    import sys

    ensure_dirs()
    logger.remove()
    if LOG_TO_CONSOLE:
        logger.add(sys.stderr, level=LOG_LEVEL, colorize=True)
    logger.add(
        LOG_FILE_PATTERN,
        level=LOG_LEVEL,
        rotation="00:00",
        retention=LOG_RETENTION,
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
    )
    _logger_ready = True
    return logger


def validate() -> list[str]:
    """启动自检，返回配置问题列表（空列表表示一切正常）。"""
    problems: list[str] = []
    cfg = get_llm_config()

    if not cfg["api_key"] and cfg["provider"] == "deepseek":
        problems.append(
            "DEEPSEEK_API_KEY 为空 —— 请在 config.py 或 .env 中填入 DeepSeek 密钥"
        )
    if cfg["provider"] == "qwen" and "127.0.0.1" in cfg["base_url"] and not LOCAL_MODE:
        problems.append(
            "QWEN_BASE_URL 仍指向本地回环地址 —— 请改成算力云上 SGLang 服务的实际地址"
        )
    if RERANK_TOP_K > VECTOR_TOP_K + BM25_TOP_K:
        problems.append("RERANK_TOP_K 大于两路召回总数，检索结果会偏少")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        problems.append("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
    return problems


if __name__ == "__main__":
    issues = validate()
    print(f"运行模式   : {'本地轻量 (LOCAL_MODE=True)' if LOCAL_MODE else '生产 (Milvus + Redis)'}")
    print(f"向量化后端 : {EMBEDDING_BACKEND}")
    print(f"当前模型   : {LLM_PROVIDER} / {get_llm_config()['model']}")
    print(f"服务地址   : {get_llm_config()['base_url']}")
    if issues:
        print("\n配置告警：")
        for item in issues:
            print(f"  - {item}")
    else:
        print("\n配置检查通过。")
