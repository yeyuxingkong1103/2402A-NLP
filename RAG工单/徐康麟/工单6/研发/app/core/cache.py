# -*- coding: utf-8 -*-
"""工单4 多级缓存与批处理工具（T7 新增；问答热路径的可观测缓存层）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

为什么要这一层（T7 任务书「其余必须实现的优化项」1/2/4）：
    * **问题归一化缓存**：把「问句 → 归一化键」的清洗结果缓存，避免每问都做正则+分词；
    * **检索结果缓存**：同一问句（同样 Top-K / 模态过滤 / 语料路由 / 配置指纹）直接返回上次候选；
    * **生成结果缓存**：同一「问句 + 证据块集合 + 提示词版本」的生成结果复用；
      三者与 qa_engine 里已落 SQLite 的**答案级缓存**配合，构成「答案 → 生成 → 检索 → 归一化」四级；
    * **批处理**：``batch_embed`` 把多条文本按 batch_size 打包成一次嵌入调用，减少往返（索引侧与图像侧复用）；
    * **可观测**：每次命中/未命中都写结构化日志与统计，报告里可直接给出命中率与省下的耗时。

设计纪律：
    * 只依赖标准库（本机 pip 通道不通，见 设计/环境事实.md §3）；
    * 线程安全（问答 Web 为多线程 ``ThreadingHTTPServer``）；
    * **TTL + LRU** 双重上限，内存可控；可选把统计与内容快照落盘到 ``研发/data/cache/``（可读 JSON，便于留痕）；
    * 缓存键**必须含配置指纹**（模型名 / Top-K / 权重 / 提示词 mtime），改算法后旧缓存自然失效，
      不允许「改了算法仍命中旧缓存」这类静默错误。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

#: 缓存默认上限（环境变量 ``RAG_CACHE__*`` 可覆盖）
DEFAULT_MAX_ENTRIES = 512
DEFAULT_TTL_S = 3600.0
#: 批处理默认批大小（嵌入后端单次可容纳的文本条数）
DEFAULT_BATCH_SIZE = 16

_WS = re.compile(r"\s+")


def _lazy_logger(logger: Any, module: str = "cache") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def normalize_for_key(text: str) -> str:
    """问句归一化（缓存键用）：全角→半角、压空白、去首尾标点、小写。

    与判分口径（``测试/common/golden.py::normalize``）**不同**：这里只求「同一问句得到同一键」，
    不做数值等价换算（数值等价是判分的事，缓存键必须精确到字面）。
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    out: list[str] = []
    for ch in raw:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:          # 全角 ASCII → 半角
            ch = chr(code - 0xFEE0)
        elif code == 0x3000:                  # 全角空格
            ch = " "
        out.append(ch)
    flat = _WS.sub(" ", "".join(out)).strip().strip("？?。.!！,，;；:：")
    return flat.lower()


def question_key(text: str) -> str:
    """归一化问句的稳定键（sha1 前 16 位）。"""
    return hashlib.sha1(normalize_for_key(text).encode("utf-8")).hexdigest()[:16]


def digest(obj: Any) -> str:
    """任意可序列化对象的稳定摘要（缓存键的一部分）。"""
    try:
        payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(obj)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def config_fingerprint(cfg: Any = None, *, extra: Mapping[str, Any] | None = None) -> str:
    """配置指纹：模型名 / Top-K / 融合权重 / 关键阈值 / 提示词 mtime。

    设计动机（T7 实测教训）：T5 曾出现「改完拒答闸门仍命中旧缓存」；
    把指纹写进缓存键后，任何影响结果的配置变化都会自动换键。
    """
    payload: dict[str, Any] = dict(extra or {})
    llm = getattr(cfg, "llm", None)
    ret = getattr(cfg, "retrieval", None)
    ans = getattr(cfg, "answer", None)
    if llm is not None:
        payload.update({"gen_model": getattr(llm, "ollama_gen_model", ""),
                        "embed_model": getattr(llm, "ollama_embed_model", ""),
                        "temperature": getattr(llm, "temperature", None),
                        "max_tokens": getattr(llm, "max_tokens", None)})
    if ret is not None:
        payload.update({"top_k": getattr(ret, "top_k", None),
                        "prompt_top_k": getattr(ret, "prompt_top_k", None),
                        "vector_k": getattr(ret, "vector_k", None),
                        "bm25_k": getattr(ret, "bm25_k", None),
                        "rrf_k": getattr(ret, "rrf_k", None)})
    if ans is not None:
        payload.update({"min_score": getattr(ans, "min_score", None),
                        "min_keyword_coverage": getattr(ans, "min_keyword_coverage", None),
                        "unknown_text": getattr(ans, "unknown_text", None)})
    try:                                       # 提示词版本（文件 mtime）
        prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
        payload["prompts"] = {p.name: int(p.stat().st_mtime) for p in sorted(prompts_dir.glob("*.txt"))}
    except OSError:
        payload["prompts"] = {}
    return digest(payload)


@dataclass(slots=True)
class CacheStats:
    """单层缓存统计（报告里逐层给出命中率）。"""

    name: str
    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    expired: int = 0
    saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hit_rate"] = self.hit_rate
        return data


class TTLCache:
    """线程安全的 LRU + TTL 缓存（值类型不限，取用方负责深拷贝语义）。"""

    def __init__(self, name: str, *, max_entries: int = DEFAULT_MAX_ENTRIES,
                 ttl_s: float = DEFAULT_TTL_S, logger: Any = None) -> None:
        self.name = name
        self.max_entries = max(1, int(max_entries))
        self.ttl_s = float(ttl_s)
        self.log = _lazy_logger(logger)
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats(name=name)

    # -- 基本操作 ---------------------------------------------------------
    def get(self, key: str) -> tuple[bool, Any]:
        """返回 ``(是否命中, 值)``；过期条目按未命中处理并计数。"""
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.stats.misses += 1
                return False, None
            stamp, value = item
            if self.ttl_s > 0 and now - stamp > self.ttl_s:
                self._data.pop(key, None)
                self.stats.expired += 1
                self.stats.misses += 1
                self.log.log_event("cache.expired", layer=self.name, key=key)
                return False, None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return True, value

    def put(self, key: str, value: Any, *, saved_ms: float = 0.0) -> None:
        """写入并维护 LRU 上限；``saved_ms`` 用于统计「缓存省下的耗时」。"""
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            self.stats.puts += 1
            if saved_ms:
                self.stats.saved_ms += float(saved_ms)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    def invalidate(self, *, prefix: str | None = None) -> int:
        """清空或按前缀清除（``config_fingerprint`` 变更后调用）。"""
        with self._lock:
            if prefix is None:
                count = len(self._data)
                self._data.clear()
            else:
                keys = [k for k in self._data if k.startswith(prefix)]
                for k in keys:
                    self._data.pop(k, None)
                count = len(keys)
        self.log.log_event("cache.invalidated", layer=self.name, removed=count, prefix=prefix)
        return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return {**self.stats.to_dict(), "entries": len(self), "max_entries": self.max_entries,
                "ttl_s": self.ttl_s}


class MultiLevelCache:
    """四级缓存门面：归一化 / 检索 / 生成 / 改写（各自独立上限与统计）。"""

    def __init__(self, *, cfg: Any = None, logger: Any = None,
                 max_entries: int | None = None, ttl_s: float | None = None) -> None:
        size = int(max_entries if max_entries is not None else _env_int("RAG_CACHE__MAX_ENTRIES", DEFAULT_MAX_ENTRIES))
        ttl = float(ttl_s if ttl_s is not None else _env_float("RAG_CACHE__TTL_S", DEFAULT_TTL_S))
        self.log = _lazy_logger(logger)
        self.enabled = str(os.environ.get("RAG_CACHE__ENABLED", "1")).strip() not in {"0", "false", "False"}
        self.fingerprint = config_fingerprint(cfg)
        self.normalized = TTLCache("normalized", max_entries=size, ttl_s=ttl, logger=self.log)
        self.retrieval = TTLCache("retrieval", max_entries=size, ttl_s=ttl, logger=self.log)
        self.generation = TTLCache("generation", max_entries=size, ttl_s=ttl, logger=self.log)
        self.rewrite = TTLCache("rewrite", max_entries=max(64, size // 4), ttl_s=ttl, logger=self.log)
        self.log.log_event("cache.ready", enabled=self.enabled, fingerprint=self.fingerprint,
                           max_entries=size, ttl_s=ttl)

    # -- 键 ---------------------------------------------------------------
    def norm_key(self, text: str) -> str:
        """归一化缓存：文本 → 归一化形式（命中即省一次正则清洗）。"""
        key = question_key(text)
        hit, value = self.normalized.get(key)
        if hit:
            return str(value)
        normalized = normalize_for_key(text)
        self.normalized.put(key, normalized)
        return normalized

    def retrieval_key(self, question: str, *, top_k: int, modality: str | None,
                      scope: str, extra: Mapping[str, Any] | None = None) -> str:
        return "r:" + digest({"v": self.fingerprint, "q": self.norm_key(question), "k": int(top_k),
                              "m": modality or "", "s": scope, "x": dict(extra or {})})

    def generation_key(self, question: str, *, chunk_ids: Sequence[str], language: str) -> str:
        return "g:" + digest({"v": self.fingerprint, "q": self.norm_key(question),
                              "c": list(chunk_ids), "lang": language})

    def rewrite_key(self, question: str, *, kind: str) -> str:
        return "w:" + digest({"v": self.fingerprint, "q": self.norm_key(question), "kind": kind})

    # -- 统计与落盘 -------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "fingerprint": self.fingerprint,
                "layers": {name: layer.to_dict() for name, layer in (
                    ("normalized", self.normalized), ("retrieval", self.retrieval),
                    ("generation", self.generation), ("rewrite", self.rewrite))}}

    def persist(self, directory: Path | str, *, logger: Any = None) -> Path:
        """把统计与键清单落盘（可读 JSON，便于留痕与报告引用）。"""
        log = _lazy_logger(logger or self.log)
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "cache_stats.json"
        payload = {"work_order": WORK_ORDER, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "directory": str(target), **self.stats()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log.log_event("cache.persisted", path=str(path), layers=len(payload["layers"]))
        return path


_CACHE_SINGLETON: MultiLevelCache | None = None
_CACHE_LOCK = threading.RLock()


def get_cache(cfg: Any = None, *, logger: Any = None, refresh: bool = False,
              max_entries: int | None = None, ttl_s: float | None = None) -> MultiLevelCache:
    """进程级缓存门面（单例）；``refresh=True`` 重建（配置变更后使用）。"""
    global _CACHE_SINGLETON
    with _CACHE_LOCK:
        if _CACHE_SINGLETON is None or refresh:
            _CACHE_SINGLETON = MultiLevelCache(cfg=cfg, logger=logger, max_entries=max_entries, ttl_s=ttl_s)
        return _CACHE_SINGLETON


# ---------------------------------------------------------------------------
# 批处理
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BatchOutcome:
    """批处理结果（含「省下的往返次数」这一可报告指标）。"""

    vectors: Any
    n_items: int = 0
    n_batches: int = 0
    batch_size: int = DEFAULT_BATCH_SIZE
    elapsed_ms: float = 0.0
    per_item_ms: float = 0.0
    saved_round_trips: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "vectors"}


def batch_embed(texts: Sequence[str], embed_fn: Callable[[Sequence[str]], Any], *,
                batch_size: int = DEFAULT_BATCH_SIZE, logger: Any = None) -> BatchOutcome:
    """把多条文本按 ``batch_size`` 打包成若干次 ``embed_fn`` 调用。

    * ``embed_fn`` 接收一个文本列表、返回可拼接的矩阵/列表（本工程为 Ollama ``/api/embed`` 的多条输入）；
    * **批内失败不放弃整批**：退化为逐条调用并把错误写入 ``errors``（显式降级，绝不静默丢数据）；
    * ``saved_round_trips = n_items - n_batches``，是「批处理优化」最直观的可报告收益。
    """
    import numpy as np

    log = _lazy_logger(logger)
    items = [str(t) for t in texts]
    size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    started = time.perf_counter()
    chunks: list[Any] = []
    errors: list[str] = []
    n_batches = 0
    for start in range(0, len(items), size):
        window = items[start:start + size]
        n_batches += 1
        try:
            chunks.append(np.asarray(embed_fn(window), dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 —— 显式降级到逐条
            errors.append(f"batch@{start}: {type(exc).__name__}: {exc}")
            for single in window:
                try:
                    chunks.append(np.asarray(embed_fn([single]), dtype=np.float32))
                except Exception as inner:  # noqa: BLE001
                    errors.append(f"single@{start}: {type(inner).__name__}: {inner}")
    matrix = np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    outcome = BatchOutcome(
        vectors=matrix, n_items=len(items), n_batches=n_batches, batch_size=size,
        elapsed_ms=elapsed_ms,
        per_item_ms=round(elapsed_ms / len(items), 3) if items else 0.0,
        saved_round_trips=max(0, len(items) - n_batches), errors=errors,
    )
    log.log_event("cache.batch_embed", **outcome.to_dict())
    return outcome


def as_rows_major(matrix: Any) -> Any:
    """把向量矩阵整理成 **C 连续 float32**（向量检索的内存布局优化）。

    依据：``vectors @ q`` 与 FAISS ``add`` 都按行主序访问；若矩阵来自 ``np.load`` 的切片/转置，
    非连续内存会让每次点积多做一次拷贝（本工程 2676×1024 规模下属于「免费优化」）。
    """
    import numpy as np

    array = np.asarray(matrix, dtype=np.float32)
    return np.ascontiguousarray(array)
