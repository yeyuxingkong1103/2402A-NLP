"""
配置中心
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

所有可调参数集中在此，读取根目录 .env。其他模块只从这里取常量，不自己读环境变量，
避免「同一参数两处各写一份」导致调参漂移。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- 工单标识
# 工单要求：代码注释需包含工单编号。所有模块统一引用此常量。
WORK_ORDER_NO = "人工智能NLP-RAG-基于PDF文档的问答系统"

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    return _get(key, str(default)).lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- 应用
APP_NAME = _get("APP_NAME", "PDF-RAG-QA")
APP_HOST = _get("APP_HOST", "0.0.0.0")
APP_PORT = _get_int("APP_PORT", 8010)
LOG_LEVEL = _get("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------- 大模型
LLM_API_KEY = _get("LLM_API_KEY")
LLM_BASE_URL = _get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = _get("LLM_MODEL", "qwen3.7-flash")
LLM_JUDGE_MODEL = _get("LLM_JUDGE_MODEL", LLM_MODEL)
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 1024)
LLM_TIMEOUT_SECONDS = _get_int("LLM_TIMEOUT_SECONDS", 60)

# ---------------------------------------------------------------- 向量模型
EMBEDDING_MODEL_PATH = _get("EMBEDDING_MODEL_PATH", r"D:\bge-small-zh-v1.5")
EMBEDDING_DIM = _get_int("EMBEDDING_DIM", 512)
EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE", 32)

# ---------------------------------------------------------------- 路径
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = ROOT_DIR / "static"
EVAL_DIR = ROOT_DIR / "data" / "eval"

PDF_PATH = ROOT_DIR / _get("PDF_PATH", "data/raw/zhaogu_yixiangshu.pdf")

# 索引产物
CHUNKS_JSONL = INDEX_DIR / "chunks.jsonl"      # 分块正文 + 元数据
EMBEDDINGS_NPY = INDEX_DIR / "embeddings.npy"  # 归一化向量矩阵 (N, D)
INDEX_META_JSON = INDEX_DIR / "index_meta.json"  # 索引元信息（模型、维度、构建时间）

# ---------------------------------------------------------------- 分块
CHUNK_SIZE = _get_int("CHUNK_SIZE", 600)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 80)
CHUNK_MIN_LENGTH = _get_int("CHUNK_MIN_LENGTH", 40)
CHUNK_MAX_LENGTH = _get_int("CHUNK_MAX_LENGTH", 1200)

# ---------------------------------------------------------------- 检索
RETRIEVAL_DENSE_TOP_K = _get_int("RETRIEVAL_DENSE_TOP_K", 20)
RETRIEVAL_SPARSE_TOP_K = _get_int("RETRIEVAL_SPARSE_TOP_K", 20)
RETRIEVAL_FINAL_TOP_K = _get_int("RETRIEVAL_FINAL_TOP_K", 6)
RETRIEVAL_BM25_BETA = _get_float("RETRIEVAL_BM25_BETA", 0.30)
RETRIEVAL_DOC_CONSENSUS_DELTA = _get_float("RETRIEVAL_DOC_CONSENSUS_DELTA", 0.25)
RETRIEVAL_MIN_EVIDENCE = _get_float("RETRIEVAL_MIN_EVIDENCE", 0.55)

# ---------------------------------------------------------------- 功能开关
QUERY_UNDERSTANDING_ENABLED = _get_bool("QUERY_UNDERSTANDING_ENABLED", True)
UPLOAD_MAX_MB = _get_int("UPLOAD_MAX_MB", 200)


def ensure_dirs() -> None:
    """确保运行期需要的目录都存在。"""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEX_DIR, UPLOAD_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def as_public_dict() -> dict:
    """对外暴露的非敏感配置（用于 /config 探针，绝不返回 API Key）。"""
    return {
        "work_order_no": WORK_ORDER_NO,
        "app_name": APP_NAME,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_configured": bool(LLM_API_KEY),
        "embedding_model_path": EMBEDDING_MODEL_PATH,
        "embedding_dim": EMBEDDING_DIM,
        "chunk": {
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "min_length": CHUNK_MIN_LENGTH,
            "max_length": CHUNK_MAX_LENGTH,
        },
        "retrieval": {
            "dense_top_k": RETRIEVAL_DENSE_TOP_K,
            "sparse_top_k": RETRIEVAL_SPARSE_TOP_K,
            "final_top_k": RETRIEVAL_FINAL_TOP_K,
            "bm25_beta": RETRIEVAL_BM25_BETA,
            "doc_consensus_delta": RETRIEVAL_DOC_CONSENSUS_DELTA,
            "min_evidence": RETRIEVAL_MIN_EVIDENCE,
        },
        "query_understanding_enabled": QUERY_UNDERSTANDING_ENABLED,
    }
