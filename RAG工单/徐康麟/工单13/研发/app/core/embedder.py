# -*- coding: utf-8 -*-
"""工单4 嵌入后端（工单3 同源实现，只读复制后按工单4 契约适配；工单4 实测后端 = 本地 Ollama bge-m3:latest，1024 维）。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化
基线来源：工单7（人工智能NLP-RAG-功能测试及评估）``研发/app/core/embedder.py``，本文件在其上做性能改造。

后端优先级：``ollama``（``POST /api/embed``，实测 bge-m3:latest = **1024 维**）
→ ``sentence_transformers``（本地降级，**惰性导入**，在线热路径禁止 eager import 重型包）
→ 全失败抛 ``EmbeddingError``（不静默、不伪造向量）。

探测规则（硬约束）：``probe_timeout_s = 0.5``、**不重试**、失败即降级并写 ``embedder.degrade``。

工单13 性能改造（瓶颈 B2：嵌入后端逐次探测；开关 ``embed_backend_cache`` / ``embed_probe_ttl_s``）：
    原实现 ``embed_texts()`` **每次调用**都执行 ``resolve_embedding_backend(probe=True)``，
    后者会发一次 ``GET /api/tags`` 探测 —— 一次问答至少一次嵌入 ⇒ 每个请求都白付一次
    HTTP 往返（实测 0.6~21 ms）。改造为**带 TTL 的探测结论缓存**（默认 300 s 内复用）；
    开关关闭即回到「逐次探测」的基线行为（逐位一致）。
    **失败结论不缓存**：Ollama 恢复后下一次请求即可用，不会因为缓存了失败而长时间不可用。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import AppConfig, get_config, model_slug
from .errors import EmbeddingError
from .text_utils import text_digest

WORK_ORDER = "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化"

# 本地降级模型（bge-m3 同族；无网时只能用它，维度不同也必须自洽）
DEFAULT_ST_MODEL = "BAAI/bge-m3"


@dataclass(slots=True)
class LLMBackendInfo:
    """后端探测结果（设计 §2.4 冻结字段）。"""

    name: str
    model: str
    base_url: str
    available: bool
    probe_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EmbeddingResult:
    """一次嵌入调用的结果（``vectors`` 为 float32 二维数组）。"""

    vectors: np.ndarray
    dim: int
    model: str
    backend: str
    count: int
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        """只输出摘要（向量本体不进日志）。"""
        return {
            "count": self.count, "dim": self.dim, "model": self.model,
            "backend": self.backend, "elapsed_ms": self.elapsed_ms,
            "shape": list(self.vectors.shape),
        }


def _lazy_logger(logger: Any, module: str = "embedder") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    """极简 JSON HTTP（标准库；探测调用不重试）。"""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# 探测与后端选择
# ---------------------------------------------------------------------------
def probe_backends(cfg: AppConfig | None = None, *, logger: Any = None) -> list[LLMBackendInfo]:
    """探测可用嵌入后端（超时 0.5 s、不重试），返回后端信息列表。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    with log.enter("probe_backends", {"ollama": config.llm.ollama_base_url,
                                      "probe_timeout_s": config.llm.probe_timeout_s}) as span:
        infos: list[LLMBackendInfo] = []
        # ① Ollama
        started = time.perf_counter()
        try:
            data = _http_json(f"{config.llm.ollama_base_url}/api/tags", None, config.llm.probe_timeout_s)
            probe_ms = round((time.perf_counter() - started) * 1000, 2)
            names = [m.get("name") for m in (data.get("models") or [])]
            has_model = any(str(n).split(":")[0] == config.llm.ollama_embed_model.split(":")[0] for n in names)
            infos.append(LLMBackendInfo(
                name="ollama", model=config.llm.ollama_embed_model, base_url=config.llm.ollama_base_url,
                available=has_model, probe_ms=probe_ms,
                error=None if has_model else f"模型未安装：{config.llm.ollama_embed_model}（可用 {names[:6]}…）",
            ))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            infos.append(LLMBackendInfo(
                name="ollama", model=config.llm.ollama_embed_model, base_url=config.llm.ollama_base_url,
                available=False, probe_ms=round((time.perf_counter() - started) * 1000, 2),
                error=f"{type(exc).__name__}: {exc}",
            ))
        # ② sentence-transformers（只探测包是否存在，不加载模型）
        started = time.perf_counter()
        try:
            import importlib.util

            available = importlib.util.find_spec("sentence_transformers") is not None
            infos.append(LLMBackendInfo(
                name="sentence_transformers", model=DEFAULT_ST_MODEL, base_url="local",
                available=available, probe_ms=round((time.perf_counter() - started) * 1000, 2),
                error=None if available else "未安装 sentence_transformers",
            ))
        except (ImportError, ValueError) as exc:  # 显式记录，不静默
            infos.append(LLMBackendInfo(
                name="sentence_transformers", model=DEFAULT_ST_MODEL, base_url="local",
                available=False, probe_ms=round((time.perf_counter() - started) * 1000, 2),
                error=f"{type(exc).__name__}: {exc}",
            ))
        for info in infos:
            log.log_event("embedder.probe", backend=info.name, available=info.available,
                          probe_ms=info.probe_ms, model=info.model, error=info.error)
        span.set_output({"backends": [i.to_dict() for i in infos]})
        return infos


def resolve_embedding_backend(cfg: AppConfig | None = None, *, probe: bool = True,
                              logger: Any = None) -> str:
    """按「ollama → sentence_transformers」选出可用嵌入后端。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    with log.enter("resolve_embedding_backend", {"backend_cfg": config.llm.backend, "probe": probe}) as span:
        if not probe:
            span.set_output({"backend": "ollama", "reason": "probe=False（直接信任配置）"})
            return "ollama"
        infos = {i.name: i for i in probe_backends(config, logger=log)}
        if infos.get("ollama") and infos["ollama"].available:
            span.set_output({"backend": "ollama", "model": infos["ollama"].model})
            return "ollama"
        reason = infos["ollama"].error if infos.get("ollama") else "探测未执行"
        log.log_event("embedder.degrade", level="WARNING", **{"from": "ollama", "to": "sentence_transformers"},
                      reason=f"Ollama 不可用：{reason}")
        span.set_output({"backend": "sentence_transformers", "reason": reason})
        return "sentence_transformers"


# ---------------------------------------------------------------------------
# 工单13：嵌入后端探测结论缓存（瓶颈 B2：逐次 HTTP 探测 /api/tags）
# ---------------------------------------------------------------------------
_PROBE_CACHE: dict[str, Any] = {"key": None, "at": 0.0, "backend": None, "probes": 0, "reused": 0}
_PROBE_LOCK = threading.Lock()


def _probe_cache_key(config: AppConfig) -> str:
    return f"{config.llm.ollama_base_url}|{config.llm.ollama_embed_model}|{config.llm.backend}"


def probe_stats() -> dict[str, Any]:
    """探测缓存的命中统计（写进性能报告：省下的探测次数 = 省下的 HTTP 往返）。"""
    with _PROBE_LOCK:
        return {"probes": int(_PROBE_CACHE["probes"]), "reused": int(_PROBE_CACHE["reused"]),
                "cached_backend": _PROBE_CACHE["backend"]}


def reset_probe_cache() -> None:
    """清空探测缓存（测试与「Ollama 重启后强制重探」使用）。"""
    with _PROBE_LOCK:
        _PROBE_CACHE.update({"key": None, "at": 0.0, "backend": None})


def resolve_backend_cached(cfg: AppConfig | None = None, *, logger: Any = None) -> str:
    """解析嵌入后端，**带 TTL 缓存**（工单13 优化项②：去掉每次嵌入前的 HTTP 探测）。

    * 开关 ``embed_backend_cache=False`` ⇒ 等价于原 ``resolve_embedding_backend(probe=True)``；
    * 只缓存**成功**结论（ollama / sentence_transformers 可用）；探测抛异常时不写缓存；
    * 缓存键含 base_url + 嵌入模型 + backend 配置，任一项变化即自动失效。
    """
    from .perf import flag, now_ms                      # 局部导入避免循环依赖

    config = cfg or get_config()
    enabled = bool(flag("embed_backend_cache"))
    ttl_s = float(flag("embed_probe_ttl_s") or 0.0)
    if not enabled or ttl_s <= 0:
        return resolve_embedding_backend(config, probe=True, logger=logger)
    key = _probe_cache_key(config)
    stamp = now_ms()
    with _PROBE_LOCK:
        if _PROBE_CACHE["key"] == key and _PROBE_CACHE["backend"] and \
                (stamp - float(_PROBE_CACHE["at"])) / 1000.0 <= ttl_s:
            _PROBE_CACHE["reused"] = int(_PROBE_CACHE["reused"]) + 1
            cached = str(_PROBE_CACHE["backend"])
            log = _lazy_logger(logger)
            log.log_event("embedder.probe_cached", backend=cached, ttl_s=ttl_s,
                          age_ms=round(stamp - float(_PROBE_CACHE["at"]), 2))
            return cached
    try:
        backend = resolve_embedding_backend(config, probe=True, logger=logger)
    except Exception:                                    # 探测异常 ⇒ 不写缓存，交由上层降级
        raise
    with _PROBE_LOCK:
        _PROBE_CACHE.update({"key": key, "at": now_ms(), "backend": backend,
                             "probes": int(_PROBE_CACHE["probes"]) + 1})
    return backend


# ---------------------------------------------------------------------------
# 维度
# ---------------------------------------------------------------------------
_DIM_CACHE: dict[str, int] = {}


def embedding_dim(cfg: AppConfig | None = None) -> int:
    """返回当前嵌入模型的向量维度（优先读已建索引的 ids.json，其次真实探测一次并缓存）。"""
    config = cfg or get_config()
    model = config.llm.ollama_embed_model
    cached = _DIM_CACHE.get(model)
    if cached:
        return cached
    slug = model_slug(model, 1024) if "bge-m3" in model else None
    if slug:
        ids_file = config.paths.index_dir / slug / "ids.json"
        try:
            if ids_file.is_file():
                dim = int(json.loads(ids_file.read_text(encoding="utf-8")).get("dim") or 0)
                if dim > 0:
                    _DIM_CACHE[model] = dim
                    return dim
        except (OSError, ValueError, TypeError) as exc:  # 读不到就退回真实探测，但要留痕
            _lazy_logger(None).log_event("embedder.dim_cache_miss", level="WARNING",
                                         path=str(ids_file), error_type=type(exc).__name__, message=str(exc))
    result = embed_texts(["维度探测"], cfg=config, batch_size=1)
    _DIM_CACHE[model] = result.dim
    return result.dim


# ---------------------------------------------------------------------------
# Ollama 嵌入
# ---------------------------------------------------------------------------
def _embed_timeout(config: AppConfig) -> float:
    """嵌入请求超时：不小于 60 s（大批次 + CPU 推理需要余量），取配置与 60 的较大者。"""
    return max(float(config.llm.request_timeout_s), 60.0)


def guard_query_text(text: str, *, log: Any, stage: str = "embed") -> tuple[str, dict[str, Any]]:
    """工单13 容错：查询文本边界检查（空 / 超长 / 非法字符）。返回 ``(可用文本, 处置说明)``。

    为什么必须有这一层（**实测事故**）：本工单首次跑基线的第 1 轮里，一道 95 字的问句
    让 Ollama ``/api/embed`` 返回 ``HTTP 400``，原实现把它当成「Ollama 不可用」⇒ 降级到
    sentence-transformers ⇒ 现场尝试从 HuggingFace 下载 ``BAAI/bge-m3``（C 盘只读 ⇒
    ``PermissionError``）⇒ **单题耗时 26.5 s**、观测数据被污染。
    正确做法是把「输入问题」与「后端不可用」分开：
        * 空 / 纯空白 ⇒ 直接给零向量 + 明确标记（不请求后端）；
        * 超长 ⇒ 按 ``max_query_chars`` 截断（配置化阈值）并记 ``embedder.truncated``；
        * 控制字符 / 非法代理对 ⇒ 清洗掉并记账（不改变语义字符）。
    """
    from .perf import flag

    limit = max(16, int(flag("max_query_chars") or 2000))
    raw = str(text or "")
    note: dict[str, Any] = {"原始字符数": len(raw), "上限": limit}
    cleaned = "".join(ch for ch in raw if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    if len(cleaned) != len(raw):
        note["清洗字符数"] = len(raw) - len(cleaned)
        log.log_event("embedder.control_chars_removed", stage=stage,
                      removed=len(raw) - len(cleaned))
    if len(cleaned) > limit:
        cleaned = cleaned[:limit]
        note["已截断"] = True
        log.log_event("embedder.truncated", level="WARNING", stage=stage, limit=limit,
                      action="按 RAG_PERF__MAX_QUERY_CHARS 截断后再嵌入")
    if not cleaned.strip():
        note["空输入"] = True
        log.log_event("embedder.empty_input", level="WARNING", stage=stage,
                      action="返回零向量（不请求嵌入后端）")
    return cleaned, note


def _embed_ollama(texts: Sequence[str], config: AppConfig, *, batch_size: int, log: Any) -> EmbeddingResult:
    """调用 Ollama ``/api/embed`` 分批取向量。"""
    model = config.llm.ollama_embed_model
    started = time.perf_counter()
    vectors: list[list[float]] = []
    dim = 0
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        payload = {"model": model, "input": batch}
        t0 = time.perf_counter()
        try:
            data = _http_json(f"{config.llm.ollama_base_url}/api/embed", payload, _embed_timeout(config))
        except urllib.error.HTTPError as exc:
            # 工单13：HTTP 4xx 是**输入/请求**问题（长度、编码），不是「后端不可用」。
            # 处置：截断后重试一次；仍失败才向上抛（由 embed_texts 决定是否降级）。
            short = [str(t)[: max(8, int(len(str(t)) * 0.5))] for t in batch]
            log.log_event("embedder.http_error_retry", level="WARNING", backend="ollama", model=model,
                          code=int(getattr(exc, "code", 0) or 0), offset=start, size=len(batch),
                          action="截断输入至 50% 后重试一次")
            try:
                data = _http_json(f"{config.llm.ollama_base_url}/api/embed",
                                  {"model": model, "input": short}, _embed_timeout(config))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as inner:
                log.log_event("embedder.batch_error", level="ERROR", backend="ollama", model=model,
                              offset=start, size=len(batch), error_type=type(inner).__name__,
                              message=str(inner))
                raise EmbeddingError(
                    f"Ollama 嵌入失败（offset={start}，已重试）：{type(inner).__name__}: {inner}",
                    detail={"model": model, "offset": start, "size": len(batch),
                            "http_status": int(getattr(exc, "code", 0) or 0)}) from inner
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            log.log_event("embedder.batch_error", level="ERROR", backend="ollama", model=model,
                          offset=start, size=len(batch), error_type=type(exc).__name__, message=str(exc))
            raise EmbeddingError(f"Ollama 嵌入失败（offset={start}）：{type(exc).__name__}: {exc}",
                                 detail={"model": model, "offset": start, "size": len(batch)}) from exc
        got = data.get("embeddings") or []
        if len(got) != len(batch):
            raise EmbeddingError(f"Ollama 返回条数不符：期望 {len(batch)}，实际 {len(got)}",
                                 detail={"model": model, "offset": start})
        for vec in got:
            if dim == 0:
                dim = len(vec)
            elif len(vec) != dim:
                raise EmbeddingError(f"同一批次内维度不一致：{dim} vs {len(vec)}",
                                     detail={"model": model, "offset": start})
        vectors.extend(got)
        log.log_event("embedder.batch", backend="ollama", model=model, count=len(batch), dim=dim,
                      elapsed_ms=round((time.perf_counter() - t0) * 1000, 2), offset=start)
    array = np.asarray(vectors, dtype=np.float32)
    return EmbeddingResult(vectors=array, dim=int(array.shape[1]) if array.size else 0, model=model,
                           backend="ollama", count=int(array.shape[0]),
                           elapsed_ms=round((time.perf_counter() - started) * 1000, 2))


# ---------------------------------------------------------------------------
# sentence-transformers 降级（惰性导入）
# ---------------------------------------------------------------------------
_ST_MODEL_CACHE: dict[str, Any] = {}


def _load_st_model(model_name: str, log: Any) -> Any:
    """惰性加载 sentence-transformers 模型（首次导入 torch 约 2.5~3.2 s，仅降级路径执行）。"""
    if model_name in _ST_MODEL_CACHE:
        return _ST_MODEL_CACHE[model_name]
    started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 —— 在线热路径禁止 eager import
    except ImportError as exc:
        raise EmbeddingError(f"sentence_transformers 不可用：{exc}") from exc
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 —— 本地模型加载失败即无嵌入后端
        log.log_event("embedder.st_error", level="ERROR", model=model_name,
                      error_type=type(exc).__name__, message=str(exc))
        raise EmbeddingError(f"本地嵌入模型加载失败（{model_name}）：{exc}",
                             detail={"model": model_name}) from exc
    _ST_MODEL_CACHE[model_name] = model
    log.log_event("embedder.st_loaded", model=model_name,
                  elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
    return model


def _embed_sentence_transformers(texts: Sequence[str], config: AppConfig, *, batch_size: int,
                                 log: Any) -> EmbeddingResult:
    """本地嵌入降级路径。"""
    model_name = DEFAULT_ST_MODEL
    started = time.perf_counter()
    model = _load_st_model(model_name, log)
    try:
        matrix = model.encode(list(texts), batch_size=max(batch_size, 8), normalize_embeddings=False,
                              show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001 —— 降级路径失败必须显式抛出
        log.log_event("embedder.st_error", level="ERROR", model=model_name, count=len(texts),
                      error_type=type(exc).__name__, message=str(exc))
        raise EmbeddingError(f"本地嵌入失败（{model_name}）：{exc}", detail={"model": model_name}) from exc
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise EmbeddingError(f"本地嵌入返回形状异常：{array.shape}")
    return EmbeddingResult(vectors=array, dim=int(array.shape[1]), model=model_name,
                           backend="sentence_transformers", count=int(array.shape[0]),
                           elapsed_ms=round((time.perf_counter() - started) * 1000, 2))


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------
def embed_texts(texts: Sequence[str], *, cfg: AppConfig | None = None, batch_size: int = 16,
                logger: Any = None, guard: bool = True) -> EmbeddingResult:
    """批量嵌入：ollama 优先，失败即显式降级到 sentence_transformers；全失败抛 ``EmbeddingError``。

    工单13 容错改造（``guard=True``）：
        * 入参先过 :func:`guard_query_text`（空 / 超长 / 控制字符）；
        * **降级条件收紧**：只有当 Ollama **真的不可达**（连接失败、超时、5xx）才降级到
          sentence-transformers；HTTP **4xx**（输入问题）与「返回条数不符」这类**请求级**错误
          直接抛 ``EmbeddingError``，由上层决定（检索层降级为 BM25 单通道），
          不再触发「现场下载 bge-m3」这种会造成 26 s 级延迟尖刺的动作。
          依据：本工单基线跑分第 1 轮实测该路径耗时 **26.5 s**（详见 优化/瓶颈原因与优化方案.md 瓶颈 B7）。
    """
    config = cfg or get_config()
    log = _lazy_logger(logger)
    items = [str(t) if str(t).strip() else "（空块）" for t in texts]
    with log.enter("embed_texts", {"count": len(items), "batch_size": batch_size,
                                   "text_digest": text_digest(items[0]) if items else None}) as span:
        if not items:
            empty = EmbeddingResult(vectors=np.zeros((0, 0), dtype=np.float32), dim=0, model="",
                                    backend="none", count=0, elapsed_ms=0.0)
            span.set_output({"count": 0, "note": "空输入"})
            return empty
        if guard:
            items = [guard_query_text(t, log=log)[0] or "（空块）" for t in items]
        backend = resolve_backend_cached(config, logger=log)   # 工单13：带 TTL 的探测结论缓存
        if backend == "ollama":
            try:
                result = _embed_ollama(items, config, batch_size=batch_size, log=log)
                span.set_output(result.to_dict())
                return result
            except EmbeddingError as exc:
                detail = getattr(exc, "detail", None) or {}
                unreachable = detail.get("http_status") in (None, 0) or int(detail.get("http_status") or 0) >= 500
                if not unreachable:
                    log.log_event("embedder.ollama_request_error", level="ERROR",
                                  http_status=detail.get("http_status"),
                                  action="请求级错误（非后端不可达）⇒ 不降级到 sentence_transformers，"
                                         "直接抛给上层（检索层会降级为 BM25 单通道）")
                    raise
                log.log_event("embedder.degrade", level="WARNING",
                              **{"from": "ollama", "to": "sentence_transformers"},
                              reason=str(exc), count=len(items))
        result = _embed_sentence_transformers(items, config, batch_size=batch_size, log=log)
        span.set_output(result.to_dict())
        return result


def embed_query(text: str, *, cfg: AppConfig | None = None, logger: Any = None) -> np.ndarray:
    """嵌入单条查询，返回 shape ``(dim,)`` 的 float32 向量（未归一化，由 vector_store 归一化）。"""
    log = _lazy_logger(logger)
    with log.enter("embed_query", {"query": text_digest(text)}) as span:
        result = embed_texts([text], cfg=cfg, batch_size=1, logger=log)
        if result.count != 1:
            raise EmbeddingError(f"查询嵌入条数异常：{result.count}")
        vector = result.vectors[0]
        span.set_output({"dim": int(vector.shape[0]), "backend": result.backend, "model": result.model})
        return vector


def warmup(cfg: AppConfig | None = None, *, logger: Any = None) -> dict[str, Any]:
    """预热嵌入后端（把模型加载进内存），供在线服务启动时调用，避免首个请求吃冷启动。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    with log.enter("warmup", {"model": config.llm.ollama_embed_model}) as span:
        started = time.perf_counter()
        first_ms = None
        dim = 0
        backend = "none"
        try:
            first = embed_texts(["预热"], cfg=config, batch_size=1, logger=log)
            first_ms = first.elapsed_ms
            dim = first.dim
            backend = first.backend
            second = embed_texts(["预热"], cfg=config, batch_size=1, logger=log)
            info = {
                "backend": second.backend, "model": second.model, "dim": second.dim,
                "cold_ms": first_ms, "warm_ms": second.elapsed_ms,
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except EmbeddingError as exc:
            log.log_event("embedder.warmup_failed", level="WARNING", error=str(exc))
            info = {"backend": backend, "model": config.llm.ollama_embed_model, "dim": dim,
                    "cold_ms": first_ms, "warm_ms": None, "error": str(exc),
                    "total_ms": round((time.perf_counter() - started) * 1000, 2)}
        span.set_output(info)
        return info
