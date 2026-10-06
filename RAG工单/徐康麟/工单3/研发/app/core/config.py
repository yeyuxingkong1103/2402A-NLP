# -*- coding: utf-8 -*-
"""工单3 配置与语料发现（设计/接口设计.md §2.1、§3.1、§8 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

职责：
    * 冻结的 dataclass 配置树（PathsConfig / LLMSettings / RetrievalSettings /
      ChunkSettings / AnswerSettings / AppConfig）与语料描述（PdfSource）；
    * ``discover_pdfs``：**自动发现** ``data/raw/*.pdf``（大小写扩展名去重、按文件名排序、
      计算 SHA256 前 16 位、试开 PDF 判定可读性）—— 严禁硬编码文件名；
    * 环境变量覆盖（``RAG_<段>__<字段>``），未识别的 ``RAG_*`` 变量写 WARN 而不静默忽略；
    * 路径工具：``file_sha256_16`` / ``model_slug``。

依赖纪律：本模块只 import 标准库与 ``pymupdf``；**禁止** eager import torch/transformers。
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 仓库根：研发/app/core/config.py → parents[3] = 工单3
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR: Path = REPO_ROOT / "研发" / "data" / "raw"
DEFAULT_PROCESSED_DIR: Path = REPO_ROOT / "研发" / "data" / "processed"
DEFAULT_INDEX_DIR: Path = REPO_ROOT / "研发" / "data" / "index"
DEFAULT_LOG_DIR: Path = REPO_ROOT / "部署" / "日志"

# 已识别的 RAG_* 环境变量（未列出的 RAG_* 会写 WARN）；run_py.ps1 用的三个变量不算未知
KNOWN_ENV_KEYS: frozenset[str] = frozenset(
    {
        "RAG_SCHEDULER_PYTHON",
        "RAG_PY_CWD",
        "RAG_PY_ECHO",
        "RAG_DATA__RAW_DIR",
        "RAG_DATA__PROCESSED_DIR",
        "RAG_DATA__INDEX_DIR",
        "RAG_LOG__DIR",
        "RAG_LOG__LEVEL",
        "RAG_LLM__BACKEND",
        "RAG_LLM__OLLAMA_BASE_URL",
        "RAG_LLM__OLLAMA_GEN_MODEL",
        "RAG_LLM__OLLAMA_STRONG_MODEL",
        "RAG_LLM__OLLAMA_EMBED_MODEL",
        "RAG_LLM__OPENAI_BASE_URL",
        "RAG_LLM__OPENAI_API_KEY",
        "RAG_LLM__OPENAI_MODEL",
        "RAG_LLM__PROBE_TIMEOUT_S",
        "RAG_LLM__REQUEST_TIMEOUT_S",
        "RAG_LLM__MAX_TOKENS",
        "RAG_LLM__TEMPERATURE",
        "RAG_LLM__ENABLE_THINK",
        "RAG_RETRIEVAL__TOP_K",
        "RAG_RETRIEVAL__VECTOR_K",
        "RAG_RETRIEVAL__BM25_K",
        "RAG_RETRIEVAL__RRF_K",
        "RAG_RETRIEVAL__TABLE_BOOST",
        "RAG_RETRIEVAL__NUMERIC_BOOST",
        "RAG_RETRIEVAL__KEYWORD_BOOST",
        "RAG_RETRIEVAL__ENABLE_LLM_RERANK",
        "RAG_RETRIEVAL__RANK_RESCUE_WEIGHT",
        "RAG_RETRIEVAL__VECTOR_WEIGHT",
        "RAG_RETRIEVAL__BM25_WEIGHT",
        "RAG_RETRIEVAL__NUMERIC_BOOST_REQUIRE_NUMERIC_QUERY",
        "RAG_CHUNK__SIZE",
        "RAG_CHUNK__OVERLAP",
        "RAG_CHUNK__MIN_SIZE",
        "RAG_CHUNK__MAX_TABLE_CHARS",
        "RAG_ANSWER__MIN_SCORE",
        "RAG_ANSWER__MIN_KEYWORD_COVERAGE",
        "RAG_ANSWER__MIN_VECTOR_SIMILARITY_EN",
        "RAG_ANSWER__UNKNOWN_TEXT",
        "RAG_ANSWER__HISTORY_TURNS",
        "RAG_ANSWER__ENABLE_SUBJECT_GATE",
        "RAG_SERVER__HOST",
        "RAG_SERVER__PORT",
        "RAG_TABLE__MIN_ROWS",
        "RAG_TABLE__MIN_COLS",
        "RAG_TABLE__MAX_TITLE_LEN",
        "RAG_VECTOR_STORE__USE_FAISS",
        "RAG_EMBEDDING__BATCH_SIZE",
        "RAG_RUN_ID",
    }
)


# ---------------------------------------------------------------------------
# 数据结构（设计 §2.1 冻结）
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class PdfSource:
    """一份待解析的 PDF 语料（file_name 全局主键，不得硬编码）。"""

    file_name: str
    path: Path
    size_bytes: int
    sha256_16: str
    page_count: int | None = None
    readable: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PdfSource":
        return cls(
            file_name=data["file_name"],
            path=Path(data["path"]),
            size_bytes=int(data["size_bytes"]),
            sha256_16=data["sha256_16"],
            page_count=data.get("page_count"),
            readable=bool(data.get("readable", False)),
        )


@dataclass(slots=True, frozen=True)
class PathsConfig:
    repo_root: Path
    raw_dir: Path
    processed_dir: Path
    index_dir: Path
    log_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass(slots=True, frozen=True)
class LLMSettings:
    backend: str = "auto"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_gen_model: str = "qwen2.5:3b"
    # 可选兜底强模型：留空 = 关闭（默认）。实测（§14.7）：打开 deepseek-r1:7b 后单题总耗时 30~145 s、
    # 首字 4.0~4.8 s → **破坏「首字 ≤3 s」硬指标**，且 14 题语义正确数未提升（9→8），故默认关闭；
    # 需要更高答案完整度时可用 RAG_LLM__OLLAMA_STRONG_MODEL=deepseek-r1:7b 临时打开。
    ollama_strong_model: str = ""
    ollama_embed_model: str = "bge-m3:latest"
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    probe_timeout_s: float = 0.5
    request_timeout_s: float = 20.0
    max_tokens: int = 512
    temperature: float = 0.2
    # 推理模型（deepseek-r1 等）默认先输出思维链，会占满 num_predict 导致正文为空；
    # False 时向 Ollama 传 think=false，要求直接作答（T6 实测项，见 部署/配置/环境事实.md §14）
    enable_think: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["openai_api_key"] = mask_secret(self.openai_api_key)
        return data


@dataclass(slots=True, frozen=True)
class RetrievalSettings:
    top_k: int = 5
    vector_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    table_boost: float = 1.25
    numeric_boost: float = 1.15
    keyword_boost: float = 1.10
    enable_llm_rerank: bool = False
    rank_rescue_weight: float = 1.0   # 单路高分救援权重（0=纯 RRF；T5 实测加此值可救回单路命中）
    numeric_boost_require_numeric_query: bool = True   # 数字加权仅对「问数字」的问题生效（T5 实测题 34/207/260 需要）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ChunkSettings:
    size: int = 500
    overlap: int = 80
    min_size: int = 120
    max_table_chars: int = 4000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class AnswerSettings:
    min_score: float = 0.12            # **T6 标定**：正样本 min 0.153 / 负样本 max 0.205 → 只作退化底线
    min_keyword_coverage: float = 0.45   # **T6 标定**：正样本 min 0.615 / 负样本 max 0.400 → 主判据（实测 14/14 通过、5/5 无关拒绝）
    # t21（§24）：**英文问句**的向量相似度底线（仅当问句没有任何专名时启用；中文路径不使用该值）。
    # 实测（bge-m3，top-3 最大向量分）：英文正样本 0.52~0.57；跨语言无关问句 0.40~0.42。
    min_vector_similarity_en: float = 0.45
    unknown_text: str = "不清楚"
    history_turns: int = 5
    enable_subject_gate: bool = True   # 主体类型闸门（题3/题4 同页互污染的通用解法）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TableSettings:
    """表格归一化阈值（环境变量 RAG_TABLE__*）。"""

    min_rows: int = 2
    min_cols: int = 2
    max_title_len: int = 60

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class VectorStoreSettings:
    """向量库设置（环境变量 ``RAG_VECTOR_STORE__*``）。"""

    use_faiss: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class EmbeddingSettings:
    """嵌入调用设置（环境变量 ``RAG_EMBEDDING__*``）。"""

    batch_size: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class AppConfig:
    paths: PathsConfig
    llm: LLMSettings
    retrieval: RetrievalSettings
    chunk: ChunkSettings
    answer: AnswerSettings
    server_host: str = "127.0.0.1"
    server_port: int = 8600
    log_level: str = "INFO"
    run_id: str = ""
    table: TableSettings = field(default_factory=TableSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": self.paths.to_dict(),
            "llm": self.llm.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "chunk": self.chunk.to_dict(),
            "answer": self.answer.to_dict(),
            "table": self.table.to_dict(),
            "vector_store": self.vector_store.to_dict(),
            "embedding": self.embedding.to_dict(),
            "server_host": self.server_host,
            "server_port": self.server_port,
            "log_level": self.log_level,
            "run_id": self.run_id,
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def mask_secret(secret: str) -> str:
    """密钥脱敏：保留前 4 后 4（长度不足 12 时整体打码）。"""
    if not secret:
        return ""
    if len(secret) < 12:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * 8}{secret[-4:]}"


def file_sha256_16(path: Path | str) -> str:
    """流式计算文件 SHA256 的前 16 位大写十六进制（1 MB 分块）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16].upper()


def model_slug(model: str, dim: int) -> str:
    """模型名归一化：``"bge-m3:latest", 1024`` → ``"bge-m3_1024"``（索引目录名）。"""
    name = str(model or "").strip()
    tag = ""
    if ":" in name:
        name, tag = name.split(":", 1)
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
    if tag and tag.lower() not in {"latest", ""}:
        slug = f"{slug}-{re.sub(r'[^0-9A-Za-z._-]+', '_', tag).strip('_')}"
    return f"{slug}_{int(dim)}"


def _lazy_logger(logger: Any, module: str) -> Any:
    """延迟获取 logger，避免 config ↔ logging_conf 循环导入。"""
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


# ---------------------------------------------------------------------------
# 语料发现（设计 §3.1）
# ---------------------------------------------------------------------------
def discover_pdfs(raw_dir: Path | str | None = None, *, logger: Any = None) -> list[PdfSource]:
    """自动发现 raw_dir 下的 PDF（``*.pdf`` + ``*.PDF`` 去重、按文件名排序）。

    * 计算 ``size_bytes`` 与 ``sha256_16``；
    * 用 ``pymupdf.open`` 试开一次填 ``page_count`` / ``readable``，随后立即 close（不泄漏句柄）；
    * 同名文件只保留第一个并写 ``config.duplicate_file`` WARN；
    * 空目录返回 ``[]`` 并写 WARN（是否致命由调用方决定）。
    """
    log = _lazy_logger(logger, "config")
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    with log.enter("discover_pdfs", {"raw_dir": str(directory)}) as span:
        if not directory.is_dir():
            raise ConfigError(f"语料目录不存在：{directory}", code="RAG-1001", detail={"raw_dir": str(directory)})

        found: dict[str, Path] = {}
        for pattern in ("*.pdf", "*.PDF"):
            for path in sorted(directory.glob(pattern)):
                if not path.is_file():
                    continue
                if path.name in found:
                    # 大小写扩展名重复（a.PDF 与 a.pdf）→ 显式告警，不静默
                    log.log_event("config.duplicate_file", level="WARNING", file_name=path.name,
                                  kept=str(found[path.name]), dropped=str(path))
                    continue
                found[path.name] = path

        sources: list[PdfSource] = []
        for name in sorted(found):
            path = found[name]
            size = path.stat().st_size
            sha = file_sha256_16(path)
            page_count: int | None = None
            readable = False
            try:
                import pymupdf  # 局部导入：仅此函数需要

                doc = pymupdf.open(str(path))
                try:
                    page_count = int(doc.page_count)
                    readable = page_count > 0
                finally:
                    doc.close()
            except Exception as exc:  # noqa: BLE001 —— 不可读文件不得静默跳过
                readable = False
                log.log_event(
                    "config.pdf_unreadable", level="WARNING", file_name=name,
                    error_type=type(exc).__name__, message=str(exc), path=str(path),
                )
            sources.append(
                PdfSource(
                    file_name=name, path=path, size_bytes=size, sha256_16=sha,
                    page_count=page_count, readable=readable,
                )
            )

        if not sources:
            log.log_event("config.discover_done", level="WARNING", raw_dir=str(directory), count=0, files=[])
        else:
            log.log_event(
                "config.discover_done", raw_dir=str(directory), count=len(sources),
                files=[s.to_dict() for s in sources],
            )
        span.set_output({"count": len(sources), "readable": sum(1 for s in sources if s.readable)})
        return sources


# ---------------------------------------------------------------------------
# 环境变量 → AppConfig（设计 §8）
# ---------------------------------------------------------------------------
def _env_get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None:
        return default
    value = str(value).strip()
    return value if value != "" else default


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ConfigError(f"环境变量 {key} 需要整数，实际为 {raw!r}", code="RAG-1002", detail={"key": key, "raw": raw}) from exc


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ConfigError(f"环境变量 {key} 需要浮点数，实际为 {raw!r}", code="RAG-1002", detail={"key": key, "raw": raw}) from exc


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    raise ConfigError(f"环境变量 {key} 需要布尔值，实际为 {raw!r}", code="RAG-1002", detail={"key": key, "raw": raw})


def _default_run_id(now: datetime | None = None) -> str:
    """运行号：``rYYYYMMDD-HHMMSS``（设计日志示例格式）。"""
    moment = now or datetime.now()
    return f"r{moment.strftime('%Y%m%d-%H%M%S')}"


def load_config(
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """按环境变量 + 内置默认值构造 AppConfig；未识别的 ``RAG_*`` 写 WARN。"""
    environ: Mapping[str, str] = env if env is not None else os.environ
    log = None
    try:
        log = _lazy_logger(None, "config")
    except Exception:  # noqa: BLE001 —— 日志尚未就绪时不得阻断配置加载（显式降级）
        log = None

    unknown = sorted(k for k in environ if k.startswith("RAG_") and k not in KNOWN_ENV_KEYS)
    if unknown and log is not None:
        log.log_event("config.unknown_env", level="WARNING", keys=unknown, count=len(unknown))

    paths = PathsConfig(
        repo_root=REPO_ROOT,
        raw_dir=Path(_env_get(environ, "RAG_DATA__RAW_DIR", str(DEFAULT_RAW_DIR))),
        processed_dir=Path(_env_get(environ, "RAG_DATA__PROCESSED_DIR", str(DEFAULT_PROCESSED_DIR))),
        index_dir=Path(_env_get(environ, "RAG_DATA__INDEX_DIR", str(DEFAULT_INDEX_DIR))),
        log_dir=Path(_env_get(environ, "RAG_LOG__DIR", str(DEFAULT_LOG_DIR))),
    )
    llm = LLMSettings(
        backend=_env_get(environ, "RAG_LLM__BACKEND", "auto"),
        ollama_base_url=_env_get(environ, "RAG_LLM__OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_gen_model=_env_get(environ, "RAG_LLM__OLLAMA_GEN_MODEL", "qwen2.5:3b"),
        ollama_strong_model=_env_get(environ, "RAG_LLM__OLLAMA_STRONG_MODEL", ""),
        ollama_embed_model=_env_get(environ, "RAG_LLM__OLLAMA_EMBED_MODEL", "bge-m3:latest"),
        openai_base_url=_env_get(environ, "RAG_LLM__OPENAI_BASE_URL", ""),
        openai_api_key=_env_get(environ, "RAG_LLM__OPENAI_API_KEY", ""),
        openai_model=_env_get(environ, "RAG_LLM__OPENAI_MODEL", ""),
        probe_timeout_s=_env_float(environ, "RAG_LLM__PROBE_TIMEOUT_S", 0.5),
        request_timeout_s=_env_float(environ, "RAG_LLM__REQUEST_TIMEOUT_S", 20.0),
        max_tokens=_env_int(environ, "RAG_LLM__MAX_TOKENS", 512),
        temperature=_env_float(environ, "RAG_LLM__TEMPERATURE", 0.2),
        enable_think=_env_bool(environ, "RAG_LLM__ENABLE_THINK", True),
    )
    retrieval = RetrievalSettings(
        top_k=_env_int(environ, "RAG_RETRIEVAL__TOP_K", 5),
        vector_k=_env_int(environ, "RAG_RETRIEVAL__VECTOR_K", 20),
        bm25_k=_env_int(environ, "RAG_RETRIEVAL__BM25_K", 20),
        rrf_k=_env_int(environ, "RAG_RETRIEVAL__RRF_K", 60),
        table_boost=_env_float(environ, "RAG_RETRIEVAL__TABLE_BOOST", 1.25),
        numeric_boost=_env_float(environ, "RAG_RETRIEVAL__NUMERIC_BOOST", 1.15),
        keyword_boost=_env_float(environ, "RAG_RETRIEVAL__KEYWORD_BOOST", 1.10),
        enable_llm_rerank=_env_bool(environ, "RAG_RETRIEVAL__ENABLE_LLM_RERANK", False),
        rank_rescue_weight=_env_float(environ, "RAG_RETRIEVAL__RANK_RESCUE_WEIGHT", 1.0),
        vector_weight=_env_float(environ, "RAG_RETRIEVAL__VECTOR_WEIGHT", 1.0),
        bm25_weight=_env_float(environ, "RAG_RETRIEVAL__BM25_WEIGHT", 1.0),
        numeric_boost_require_numeric_query=_env_bool(environ, "RAG_RETRIEVAL__NUMERIC_BOOST_REQUIRE_NUMERIC_QUERY", True),
    )
    chunk = ChunkSettings(
        size=_env_int(environ, "RAG_CHUNK__SIZE", 500),
        overlap=_env_int(environ, "RAG_CHUNK__OVERLAP", 80),
        min_size=_env_int(environ, "RAG_CHUNK__MIN_SIZE", 120),
        max_table_chars=_env_int(environ, "RAG_CHUNK__MAX_TABLE_CHARS", 4000),
    )
    answer = AnswerSettings(
        min_score=_env_float(environ, "RAG_ANSWER__MIN_SCORE", 0.12),
        min_keyword_coverage=_env_float(environ, "RAG_ANSWER__MIN_KEYWORD_COVERAGE", 0.45),
        min_vector_similarity_en=_env_float(environ, "RAG_ANSWER__MIN_VECTOR_SIMILARITY_EN", 0.45),
        unknown_text=_env_get(environ, "RAG_ANSWER__UNKNOWN_TEXT", "不清楚"),
        history_turns=_env_int(environ, "RAG_ANSWER__HISTORY_TURNS", 5),
        enable_subject_gate=_env_bool(environ, "RAG_ANSWER__ENABLE_SUBJECT_GATE", True),
    )
    table = TableSettings(
        min_rows=_env_int(environ, "RAG_TABLE__MIN_ROWS", 2),
        min_cols=_env_int(environ, "RAG_TABLE__MIN_COLS", 2),
        max_title_len=_env_int(environ, "RAG_TABLE__MAX_TITLE_LEN", 60),
    )
    vector_store = VectorStoreSettings(
        use_faiss=_env_bool(environ, "RAG_VECTOR_STORE__USE_FAISS", False),
    )
    embedding = EmbeddingSettings(
        batch_size=_env_int(environ, "RAG_EMBEDDING__BATCH_SIZE", 16),
    )
    cfg = AppConfig(
        paths=paths,
        llm=llm,
        retrieval=retrieval,
        chunk=chunk,
        answer=answer,
        server_host=_env_get(environ, "RAG_SERVER__HOST", "127.0.0.1"),
        server_port=_env_int(environ, "RAG_SERVER__PORT", 8600),
        log_level=_env_get(environ, "RAG_LOG__LEVEL", "INFO").upper(),
        run_id=_env_get(environ, "RAG_RUN_ID", _default_run_id()),
        table=table,
        vector_store=vector_store,
        embedding=embedding,
    )
    # 基础校验：fail fast（不静默接受非法取值）
    if chunk.overlap >= chunk.size:
        raise ConfigError(
            f"RAG_CHUNK__OVERLAP({chunk.overlap}) 必须小于 RAG_CHUNK__SIZE({chunk.size})",
            code="RAG-1002", detail={"overlap": chunk.overlap, "size": chunk.size},
        )
    if llm.probe_timeout_s <= 0:
        raise ConfigError("RAG_LLM__PROBE_TIMEOUT_S 必须为正数", code="RAG-1002")
    if overrides:
        cfg = replace(cfg, **{k: v for k, v in overrides.items() if hasattr(cfg, k)})
    if log is not None:
        log.log_event("config.load", run_id=cfg.run_id, paths=paths.to_dict(), llm=llm.to_dict(),
                      chunk=chunk.to_dict(), retrieval=retrieval.to_dict(), answer=answer.to_dict(),
                      table=table.to_dict(), vector_store=vector_store.to_dict(),
                      embedding=embedding.to_dict(), unknown_env=unknown)
    return cfg


_CONFIG_CACHE: AppConfig | None = None


def get_config(*, refresh: bool = False) -> AppConfig:
    """进程级配置缓存（``refresh=True`` 强制重载）。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or refresh:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


def discover_issuer_names(raw_dir: Path | str | None = None, *, logger: Any = None) -> list[str]:
    """从语料**动态提取**发行人全称（**禁止硬编码公司名**）。

    做法：读每份 PDF 的前 3 页文本，用「…有限公司/股份有限公司」模式计数，取出现次数最多的前 3 个。
    结果供 ``strip_issuer_names``（主体类型闸门）使用；提取失败返回空列表（闸门 fail-open）。
    """
    log = _lazy_logger(logger, "config")
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    counter: dict[str, int] = {}
    try:
        import pymupdf
        import re as _re

        pattern = _re.compile(r"[\u4e00-\u9fff]{2,12}(?:股份)?有限公司")
        for source in discover_pdfs(directory, logger=log):
            if not source.readable:
                continue
            with pymupdf.open(str(source.path)) as doc:
                for index in range(min(3, int(doc.page_count))):
                    for name in pattern.findall(doc.load_page(index).get_text("text") or ""):
                        counter[name] = counter.get(name, 0) + 1
    except Exception as exc:  # noqa: BLE001 —— 提取失败即返回空表（闸门 fail-open），但必须留痕
        log.log_event("config.issuer_names_failed", level="WARNING",
                      error_type=type(exc).__name__, message=str(exc))
        return []
    names = [n for n, _ in sorted(counter.items(), key=lambda kv: -kv[1])[:3]]
    log.log_event("config.issuer_names", names=names, candidates=len(counter))
    return names


def describe_runtime() -> dict[str, Any]:
    """返回解释器与关键路径摘要（供 run.start 与报告留痕）。"""
    return {
        "work_order": WORK_ORDER,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "repo_root": str(REPO_ROOT),
        "raw_dir": str(DEFAULT_RAW_DIR),
        "processed_dir": str(DEFAULT_PROCESSED_DIR),
        "index_dir": str(DEFAULT_INDEX_DIR),
        "log_dir": str(DEFAULT_LOG_DIR),
    }
