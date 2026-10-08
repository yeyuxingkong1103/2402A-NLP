# -*- coding: utf-8 -*-
"""rag2 —— RAG_2 统一离线包。

把 RAG_2 下所有可调用模块聚合为统一入口 ``import rag2``，并新增
SQLite 离线向量库（``OfflineStore``）与端到端离线流水线（``OfflineRAG``），
无需 Milvus / MySQL 服务器即可完成「识别 → 解析 → 分块 → 向量化 → 入库 → 检索」。

子模块一览：

- ``rag2.data_type``        文件类型识别
- ``rag2.mineru_parser``    MinerU 文档解析
- ``rag2.pdf_table``        PDF 表格抽取
- ``rag2.ocr``              图片 OCR（PaddleOCR-VL）
- ``rag2.hybrid_retriever`` Milvus + BM25 混合检索 + bge 重排（在线）
- ``rag2.mysql_client``     MySQL 读写
- ``rag2.store``            SQLite 离线向量库（离线）
- ``rag2.pipeline``         离线端到端流水线（离线）
- ``rag2.ragas_eval``       RAGAS 结果评估（离线打分，接本地 Ollama + bge-m3）
- ``rag2.server``           FastAPI 在线服务（SSE 流式 + 多用户/多角色 + 网页端）
- ``rag2.online_chat``      在线 RAG 问答编排器（多轮对话 + 短期记忆）
- ``rag2.redis_memory``     Redis 短期记忆（TTL 自动过期）
- ``rag2.llm_client``       OpenAI 兼容 LLM 客户端（默认 Ollama）
- ``rag2.roles``            角色定义与加载
- ``rag2.config``           在线阶段配置加载
- ``rag2.logging_config``   统一日志系统

用法示例：

    import rag2

    rag = rag2.OfflineRAG(db_path="kb.sqlite", embed_model="D:/modelscope/bge-m3")
    rag.add_file("文档.pdf")
    for h in rag.search("问题", top_k=3):
        print(h.score, h.text)
"""

from __future__ import annotations

__version__ = "1.0.0"

# 子模块（可直接 rag2.data_type.detect(...) 访问）
from . import (  # noqa: F401
    data_type,
    device,
    mineru_parser,
    pdf_table,
    ocr,
    vlm,
    hybrid_retriever,
    mysql_client,
    store,
    pipeline,
    ragas_eval,
)

# ---- 文件类型识别 ----
from .data_type import FileInfo, detect, detect_bytes, scan_directory, summarize

# ---- MinerU 文档解析 ----
from .mineru_parser import ParseResult, MineruParser, parse_pdf, read_zip, start_server

# ---- PDF 表格 ----
from .pdf_table import (
    TableResult,
    PDFTableExtractor,
    extract_tables,
    analyze_pdf,
    tables_to_csv,
    tables_to_json,
)

# ---- OCR ----
from .ocr import OCRResult, ImageOCR, ocr_image, ocr_images, ocr_directory, get_pipeline

# ---- 多模态视觉语言模型（VLM）----
from .vlm import (
    ImageDescriber,
    describe_image,
    answer_image,
    encode_image_data_url,
    resolve_image_data_url,
    image_content_part,
    DEFAULT_PROMPT,
)

# ---- 混合检索（在线）----
from .hybrid_retriever import (
    Hit,
    FusedItem,
    BM25Index,
    HybridRetriever,
    tokenize,
    weighted_rrf,
    DEFAULT_RERANK_MODEL,
)

# ---- MySQL ----
from .mysql_client import (
    MySQLConfig,
    MySQLClient,
    connect,
    query,
    query_one,
    query_scalar,
    execute,
)

# ---- SQLite 离线库 + 离线流水线 ----
from .store import OfflineStore
from .pipeline import OfflineRAG, chunk_text

# ---- RAGAS 结果评估 ----
from .ragas_eval import (
    RagasEvaluator,
    evaluate,
    evaluate_rag,
    get_evaluator,
    clear_evaluator_cache,
    ragas_available,
    available_metrics,
    METRIC_GROUPS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_EMBED_MODEL,
)

# ---- 设备解析 ----
from .device import resolve_device, cuda_available

# ---- 统一日志 ----
from .logging_config import setup_logging, get_logger

# ---- 在线阶段配置 ----
from .config import (
    RAG2Config,
    load_config,
    get_config,
    DEFAULTS,
    EDITABLE_FIELDS,
    validate_updates,
    apply_updates,
    save_config,
)

# ---- 角色 ----
from .roles import Role, DEFAULT_ROLES, load_roles, get_role, list_roles, role_from_dict, save_roles

# ---- LLM 客户端 ----
from .llm_client import LLMClient

# ---- Redis 短期记忆 ----
from .redis_memory import RedisMemory

# ---- 在线问答编排器 ----
from .online_chat import RAGChat

# __all__ 决定 `from rag2 import *` 会导入哪些名字；这里显式列出公开 API，避免误导出内部函数。
__all__ = [
    "__version__",
    # data_type
    "FileInfo", "detect", "detect_bytes", "scan_directory", "summarize",
    # mineru_parser
    "ParseResult", "MineruParser", "parse_pdf", "read_zip", "start_server",
    # pdf_table
    "TableResult", "PDFTableExtractor", "extract_tables", "analyze_pdf",
    "tables_to_csv", "tables_to_json",
    # ocr
    "OCRResult", "ImageOCR", "ocr_image", "ocr_images", "ocr_directory", "get_pipeline",
    # vlm
    "ImageDescriber", "describe_image", "answer_image", "encode_image_data_url",
    "resolve_image_data_url", "image_content_part", "DEFAULT_PROMPT",
    # hybrid_retriever
    "Hit", "FusedItem", "BM25Index", "HybridRetriever", "tokenize", "weighted_rrf",
    "DEFAULT_RERANK_MODEL",
    # mysql_client
    "MySQLConfig", "MySQLClient", "connect", "query", "query_one", "query_scalar", "execute",
    # store / pipeline
    "OfflineStore", "OfflineRAG", "chunk_text",
    # ragas_eval
    "RagasEvaluator", "evaluate", "evaluate_rag", "get_evaluator", "clear_evaluator_cache",
    "ragas_available", "available_metrics", "METRIC_GROUPS",
    "DEFAULT_LLM_MODEL", "DEFAULT_LLM_BASE_URL", "DEFAULT_EMBED_MODEL",
    # device
    "resolve_device", "cuda_available",
    # logging
    "setup_logging", "get_logger",
    # config
    "RAG2Config", "load_config", "get_config", "DEFAULTS",
    "EDITABLE_FIELDS", "validate_updates", "apply_updates", "save_config",
    # roles
    "Role", "DEFAULT_ROLES", "load_roles", "get_role", "list_roles",
    "role_from_dict", "save_roles",
    # llm / redis / chat
    "LLMClient", "RedisMemory", "RAGChat",
    # server（惰性导出）
    "create_app",
]


def __getattr__(name: str):
    """惰性导出重量级子模块（``server`` 依赖 fastapi，避免 ``import rag2`` 强依赖）。

    注意：这里必须用 ``importlib.import_module`` 而不是 ``from . import server``。
    ``from . import server`` 会先在本包上做属性查找，而属性查找会再次触发本函数，
    从而无限递归（RecursionError）。
    """
    if name in ("create_app", "server"):
        import importlib

        module = importlib.import_module(f"{__name__}.server")
        globals()[name] = module if name == "server" else module.create_app
        return globals()[name]
    raise AttributeError(f"module 'rag2' has no attribute {name!r}")
