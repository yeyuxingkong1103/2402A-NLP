# -*- coding: utf-8 -*-
"""Ollama 嵌入后端（bge-m3）。

链路位置
--------
离线：``PDF -> 解析 -> 分块 -> 【本模块】Ollama /api/embed -> Milvus``
在线：``用户提问 -> 【本模块】Ollama /api/embed -> Milvus 相似度检索 -> LLM``

三条硬性约定
------------
1. **走 Ollama 的 ``/api/embed``（批量 ``input: [...]``）**，一次 HTTP 往返处理
   多段文本；绝不逐条请求。批大小由 ``EMBED_BATCH_SIZE`` / 构造参数控制。
2. **维度不写死**：第一次真实响应里用 ``len(embeddings[0])`` 得到维度并缓存，
   同时写进 ``embedding_dim`` 指标（t1 的 metrics，单位 dim）。
   **绝不引入 torch / sentence-transformers**——嵌入全部交给 Ollama 进程。
3. **base_url 一律取 ``config.ollama_base_url``**（默认各机自用：Windows 用本机
   RTX 2060 那份 Ollama，VM 用它自己那份），**不硬编码 VM 地址**。

嵌入缓存
--------
内容 sha256 -> 向量，存 Redis（key 前缀 ``legal_rag:embed:``，TTL 取
``EMBED_CACHE_TTL``）；Redis 不可用按配置降级为进程内 LRU。命中/未命中都计入
``embed_cache_hit_total`` / ``embed_cache_miss_total`` 指标并打日志
（INFO：批大小、命中数、耗时；DEBUG：Ollama 调用前后耗时）。

失败处理
--------
超时取 ``EMBED_TIMEOUT``；连接失败/超时/5xx/429 指数退避重试（次数取
``LLM_MAX_RETRIES``）。重试耗尽抛 :class:`~legal_rag.embedding._client.OllamaHTTPError`
（含 URL、状态码、尝试次数、修复建议），**绝不吞异常**；
维度不一致抛 :class:`EmbeddingDimError`（含期望/实际维度与模型名）。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Sequence

from .. import metrics as M
from ..observability import log_io, timed, traced, wrap_errors
from ._client import OllamaHTTPClient, OllamaHTTPError
from .base import Embedder
from .cache import EmbedCache, build_embed_cache, content_key, iter_digests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bge-m3"
#: 批大小上限（Ollama 对超大 batch 会一次性吃掉很多内存，保守上限）
MAX_BATCH = 256


class EmbeddingDimError(RuntimeError):
    """同一次响应内向量维度不一致，或与已缓存维度冲突。"""

    def __init__(self, message: str, *, expected: int = 0, actual: int = 0,
                 model: str = "") -> None:
        super().__init__(message)
        self.expected = expected
        self.actual = actual
        self.model = model


def _ollama_options(config: Any) -> dict[str, Any]:
    """把 RagConfig.ollama 子配置摘出来（容错：缺字段时用 Ollama 默认值）。"""
    cfg = getattr(config, "ollama", None)
    get = (lambda name, default: getattr(cfg, name, default)) if cfg is not None else (
        lambda name, default: default)
    return {
        "base_url": get("base_url", "http://127.0.0.1:11434"),
        "model": get("embedding_model", DEFAULT_MODEL),
        "timeout": float(get("embed_timeout", 60.0)),
        "max_retries": int(get("max_retries", 2)),
        "retry_backoff": float(get("retry_backoff", 1.0)),
        "api_key": get("api_key", ""),
        "keep_alive": get("keep_alive", ""),
        "num_gpu": get("num_gpu", None),
        "num_thread": get("num_thread", None),
    }


class OllamaEmbedder(Embedder):
    """Ollama ``/api/embed`` 嵌入后端。"""

    name = "ollama-bge-m3"

    def __init__(
        self,
        model: str = "",
        *,
        config: Any = None,
        base_url: str = "",
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        batch_size: int = 16,
        cache: EmbedCache | None = None,
        client: OllamaHTTPClient | None = None,
    ) -> None:
        opts = _ollama_options(config)
        self.config = config
        self.model = (model or opts["model"] or DEFAULT_MODEL).strip()
        self.base_url = (base_url or opts["base_url"]).rstrip("/")
        self.timeout = float(timeout if timeout is not None else opts["timeout"])
        self.max_retries = int(max_retries if max_retries is not None else opts["max_retries"])
        self.retry_backoff = float(
            retry_backoff if retry_backoff is not None else opts["retry_backoff"])
        self.keep_alive = opts["keep_alive"]
        self.num_gpu = opts["num_gpu"]
        self.num_thread = opts["num_thread"]
        self.batch_size = max(1, min(int(batch_size), MAX_BATCH))

        self.client = client or OllamaHTTPClient(
            self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            api_key=opts["api_key"],
        )
        #: 缓存：显式传入优先；否则按 config.cache 构建（Redis 优先，失败降级 LRU）
        self.cache = cache if cache is not None else (
            build_embed_cache(config) if config is not None else None)
        self._dim: int = 0
        self._dim_lock = threading.Lock()
        self.calls = 0
        self.texts_embedded = 0
        self.cache_hits = 0
        #: 上一次 ``embed_texts`` 的分叉统计（缓存命中 vs 真走 HTTP），
        #: 供 ingest 侧汇总日志直接取用，避免调用方各自去推断。
        self.last_embed_stats: dict[str, Any] = {}
        # 启动即可定位：一次调用前后的全部关键参数（含缓存是否真的生效）都落一条日志，
        # 便于把「这批为什么全走 HTTP」直接归因到缓存后端/超时/批大小上。
        logger.info(
            "嵌入后端已初始化：provider=ollama model=%s url=%s 批大小=%d 超时=%.1fs "
            "重试=%d 缓存=%s（维度待首次响应实测）",
            self.model, self.base_url, self.batch_size, self.timeout,
            self.max_retries, self._cache_label())

    # ------------------------------------------------------------------
    # 维度
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """当前嵌入维度；**首次调用前为 0**（不写死，必须实测取得）。"""
        return self._dim

    def _record_dim(self, dim: int, *, source: str) -> int:
        with self._dim_lock:
            if self._dim and self._dim != dim:
                M.counter("dim_mismatch_total").inc(model=self.model)
                raise EmbeddingDimError(
                    f"嵌入维度不一致：模型 {self.model} 本次返回 {dim} 维，"
                    f"但此前已实测为 {self._dim} 维（来源={source}）。\n"
                    f"说明：同一模型进程内不应出现两种维度；若刚换了嵌入模型，"
                    f"请同时更换 Milvus collection 名（MILVUS_COLLECTION）后重建索引。",
                    expected=self._dim, actual=dim, model=self.model,
                )
            if not self._dim:
                self._dim = int(dim)
                logger.info("实测嵌入维度 embedding_dim=%d（模型=%s，来源=%s，base_url=%s）",
                            dim, self.model, source, self.base_url)
            M.gauge("embedding_dim").set(float(self._dim), model=self.model)
            return self._dim

    def ensure_dim(self) -> int:
        """探测（或返回已缓存的）嵌入维度，用于建 Milvus collection 前校验。

        走一次真实嵌入请求——**维度只能从真实响应取得**，不靠猜。
        """
        if self._dim:
            return self._dim
        self.embed_texts(["维度探测 probe"])
        return self._dim

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------
    def _cache_lookup(self, keys: list[str]) -> dict[str, list[float]]:
        if self.cache is None:
            M.counter("embed_cache_miss_total").inc(len(keys))
            return {}
        found = self.cache.get_many(keys)
        hits = len(found)
        misses = max(len(keys) - hits, 0)
        if hits:
            M.counter("embed_cache_hit_total").inc(hits)
        if misses:
            M.counter("embed_cache_miss_total").inc(misses)
        self.cache_hits += hits
        # 全命中时不会走 Ollama，维度只能从缓存向量里取值（否则 ensure_dim 会返回 0）
        if hits and not self._dim:
            for vector in found.values():
                if vector:
                    self._record_dim(len(vector), source="cache")
                    break
        return found

    def _cache_store(self, keys: list[str], vectors: list[list[float]]) -> None:
        if self.cache is None or not vectors:
            return
        self.cache.set_many(list(zip(keys, vectors)))

    # ------------------------------------------------------------------
    # Ollama 调用
    # ------------------------------------------------------------------
    def _payload(self, texts: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        options: dict[str, Any] = {}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        if self.num_thread is not None:
            options["num_thread"] = self.num_thread
        if options:
            payload["options"] = options
        return payload

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """一次 Ollama 批量嵌入（无缓存逻辑，纯 HTTP + 解析）。"""
        payload = self._payload(texts)
        started = time.perf_counter()
        with log_io(f"embed_batch[{len(texts)}]", f"ollama:{self.base_url}",
                    operation="embed", count=len(texts),
                    meta={"model": self.model, "batch_size": len(texts)}) as io:
            try:
                data = self.client.post_json("/api/embed", payload,
                                             timeout=self.timeout, operation="ollama.embed")
            except OllamaHTTPError as exc:
                logger.error("Ollama 嵌入调用失败：%s\n上下文：model=%s url=%s 批次=%d",
                             exc, self.model, self.base_url, len(texts))
                raise
            except Exception as exc:  # noqa: BLE001 - 带上下文后继续抛，不吞
                logger.exception("Ollama 嵌入调用出现未预期异常：model=%s url=%s",
                                 self.model, self.base_url)
                raise RuntimeError(
                    f"Ollama 嵌入失败（model={self.model}, url={self.base_url}, "
                    f"批次={len(texts)}）：{type(exc).__name__}: {exc}") from exc

            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise EmbeddingDimError(
                    f"Ollama 返回的 embeddings 数量与请求不符：请求 {len(texts)} 条，"
                    f"返回 {len(embeddings) if isinstance(embeddings, list) else '非列表'} 条"
                    f"（model={self.model}, url={self.base_url}）",
                    model=self.model,
                )
            io["meta"]["ollama_total_duration_ns"] = data.get("total_duration")

        vectors: list[list[float]] = []
        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list) or not vector:
                raise EmbeddingDimError(
                    f"Ollama 返回的第 {index} 条向量为空（model={self.model}）",
                    model=self.model)
            vectors.append([float(v) for v in vector])

        elapsed = time.perf_counter() - started
        M.observe("embed_seconds", elapsed, model=self.model)
        M.observe("embed_batch_size", float(len(texts)), model=self.model)
        return vectors

    # ------------------------------------------------------------------
    # 日志辅助：把「走缓存还是走 Ollama」这个语义分叉点说清楚
    # ------------------------------------------------------------------
    def _cache_label(self) -> str:
        """缓存后端标签（含降级原因），用于汇总日志。"""
        if self.cache is None:
            return "none（缓存已关闭）"
        label = getattr(self.cache, "label", None)
        if callable(label):
            return str(label())
        return str(self.cache.name)

    @staticmethod
    def _iter_cause_chain(exc: BaseException, limit: int = 8):
        """沿 ``cause`` / ``__cause__`` 展开异常链（去重、限深）。

        为什么必须展开：``wrap_errors`` 会把底层异常包成 ``TracedError``，
        HTTP 状态码留在**内层** ``OllamaHTTPError`` 上；只看最外层会把 404
        误分类成 ``unexpected_TracedError``，线索就丢了。
        """
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen and len(seen) < limit:
            seen.add(id(current))
            yield current
            nxt = getattr(current, "cause", None) or getattr(current, "__cause__", None)
            current = nxt if isinstance(nxt, BaseException) else None

    @classmethod
    def _classify_embed_failure(cls, exc: BaseException) -> str:
        """把嵌入失败归类，便于按类别 grep 定位（不求穷尽，只求可分流）。

        ====================  ==================================================
        分类                  触发条件
        ====================  ==================================================
        ``model_not_found``   HTTP 404（多半是模型名写错 / 没 pull）
        ``rate_limited``      HTTP 429
        ``server_error``      HTTP 5xx（Ollama 自己出错）
        ``http_<code>``       其余 4xx
        ``network_unreachable`` 连不上（连接被拒 / 解析失败 / 服务未启动）
        ``timeout``           连接或读超时
        ``response_shape_mismatch`` 返回体条数/维度不对
        ``unexpected_<Exc>``  其它
        ====================  ==================================================

        分类会先沿异常链找带 HTTP 状态码的那一层（见 :meth:`_iter_cause_chain`）。
        """
        chain = list(cls._iter_cause_chain(exc))

        # 1) 先找带 HTTP 状态码的那一层（可能被 TracedError 包在内层）
        for item in chain:
            status = int(getattr(item, "status", 0) or 0)
            if not status:
                continue
            if status == 404:
                return "model_not_found"
            if status == 429:
                return "rate_limited"
            if 500 <= status < 600:
                return "server_error"
            return f"http_{status}"

        # 2) 响应体形状不对
        if any(isinstance(item, EmbeddingDimError) for item in chain):
            return "response_shape_mismatch"

        # 3) 连接/超时：看整条链的类型名与消息
        text = " ".join(f"{type(item).__name__} {item}".lower() for item in chain)
        if "timed out" in text or "timeout" in text:
            return "timeout"
        if any(token in text for token in
               ("connection", "refused", "unreachable", "resolve", "protocol", "reset")):
            return "network_unreachable"
        return f"unexpected_{type(exc).__name__}"

    # ------------------------------------------------------------------
    # Embedder 接口
    # ------------------------------------------------------------------
    @traced(operation="ollama.embed_texts", level=logging.DEBUG, log_result=False)
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入；顺序与输入严格一致，命中缓存的条目不发起 HTTP 请求。

        **日志义务（t15）**：缓存命中/未命中是「走 Redis 还是走 Ollama」的语义分叉点，
        必须留痕。因此本方法保证：

        * 每批成功/失败各一条（**按批聚合，绝不逐条**）；
        * 本方法结束时**一条汇总 INFO**，直接回答「这次到底有没有走 Ollama」；
        * 汇总里说明 ``embed_seconds`` 只统计真实 HTTP —— 全命中时样本数为 0 属设计，
          不是指标丢失（t12 就是在这里踩过坑）。
        """
        started = time.perf_counter()
        items = list(texts)
        if not items:
            self.last_embed_stats = {"total": 0, "cache_hits": 0, "cache_misses": 0,
                                     "http_batches": 0, "http_texts": 0, "model": self.model}
            logger.info("嵌入请求为空（0 条），未触碰缓存与 Ollama")
            return []

        keys = iter_digests(items, self.model)
        cached = self._cache_lookup(keys)
        hits = len(cached)
        cache_label = self._cache_label()

        # 未命中的位置（同一个 key 只请求一次，避免批内重复文本往返）
        missing_positions: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            if key not in cached:
                missing_positions.setdefault(key, []).append(index)

        missing_keys = list(missing_positions)
        misses = len(missing_keys)
        total = len(missing_keys)
        n_batches = (total + self.batch_size - 1) // self.batch_size if total else 0
        # 去重说明：同一个 key 只请求一次，因此 **命中 + 未命中 = 去重后的条数**。
        # 不写清楚的话，「共 2 条，命中 0 / 未命中 1」会让人以为少算了一条。
        unique = hits + misses
        dedup_note = "" if unique == len(items) else f"，文本去重后 {unique} 条"

        # ---- 分叉点：命中/未命中一次性说清（每请求一条，不是每 chunk 一条）----
        if total == 0:
            logger.info(
                "嵌入缓存分叉：共 %d 条%s，命中 %d / 未命中 0 —— 本请求不会调用 Ollama"
                "（缓存=%s，模型=%s）", len(items), dedup_note, hits, cache_label, self.model)
        else:
            logger.info(
                "嵌入缓存分叉：共 %d 条%s，命中 %d / 未命中 %d（真正走 Ollama 的条数；"
                "命中+未命中=去重后条数），将分 %d 批发起 HTTP"
                "（缓存=%s，模型=%s，批大小=%d）",
                len(items), dedup_note, hits, misses, n_batches, cache_label, self.model,
                self.batch_size)

        http_texts_done = 0
        for batch_index, start in enumerate(range(0, total, self.batch_size), start=1):
            batch_keys = missing_keys[start:start + self.batch_size]
            batch_texts = [items[missing_positions[key][0]] for key in batch_keys]
            logger.info("[BEFORE] Ollama /api/embed 第 %d/%d 批：%d 条（模型=%s，url=%s）",
                        batch_index, n_batches, len(batch_texts), self.model, self.base_url)
            batch_started = time.perf_counter()
            try:
                # 注：log_io/wrap_errors 已负责「调用前后 + 异常上下文」，
                # 这里额外补一条**按类分流**的失败日志，便于 grep 定位。
                with wrap_errors("ollama.embed_texts", model=self.model, url=self.base_url,
                                 batch=len(batch_texts), offset=start):
                    batch_vectors = self._embed_batch(batch_texts)
            except Exception as exc:  # noqa: BLE001 - 记录后原样抛出，绝不吞
                logger.error(
                    "Ollama 嵌入批次失败：第 %d/%d 批，批量=%d 条，原因分类=%s，"
                    "模型=%s，url=%s，已完成 %d/%d 条（本请求命中 %d / 未命中 %d）：%s",
                    batch_index, n_batches, len(batch_texts),
                    self._classify_embed_failure(exc), self.model, self.base_url,
                    http_texts_done, total, hits, misses, exc)
                self.last_embed_stats = {
                    "model": self.model, "total": len(items),
                    "cache_hits": hits, "cache_misses": misses,
                    "http_batches": batch_index - 1, "http_texts": http_texts_done,
                    "http_failed_batch_size": len(batch_texts),
                    "failure_class": self._classify_embed_failure(exc),
                    "elapsed": time.perf_counter() - started,
                    "cache_backend": cache_label, "ok": False,
                }
                raise
            batch_elapsed = time.perf_counter() - batch_started
            for key, vector in zip(batch_keys, batch_vectors):
                cached[key] = vector
            self._record_dim(len(batch_vectors[0]), source="embed_texts")
            self._cache_store(batch_keys, batch_vectors)
            self.calls += 1
            self.texts_embedded += len(batch_texts)
            http_texts_done += len(batch_texts)
            logger.info(
                "[AFTER] Ollama /api/embed 第 %d/%d 批完成：本批 %d 条，"
                "累计走 HTTP %d/%d 条（本请求 命中 %d / 未命中 %d）；"
                "模型=%s 缓存=%s 耗时=%.1fms",
                batch_index, n_batches, len(batch_texts), http_texts_done, total,
                hits, misses, self.model, cache_label, batch_elapsed * 1000.0)

        try:
            result = [cached[key] for key in keys]
        except KeyError as exc:  # pragma: no cover - 理论上不可达
            raise RuntimeError(
                f"嵌入结果装配失败：缺少 key {exc}（model={self.model}, 条数={len(items)}）"
            ) from exc

        # 最终一致性校验：所有向量维度必须一致（缓存脏数据也会在这里被抓住）
        dims = {len(vector) for vector in result}
        if len(dims) > 1:
            raise EmbeddingDimError(
                f"嵌入结果维度不一致：本次 {len(result)} 条向量出现维度 {sorted(dims)}"
                f"（model={self.model}）。\n"
                f"提示：可能是嵌入缓存里混入了别的模型/维度的脏数据，"
                f"可清掉 legal_rag:embed:* 后重试。",
                expected=int(self._dim), actual=int(max(dims)), model=self.model)
        if dims:
            self._record_dim(dims.pop(), source="result_verify")

        elapsed = time.perf_counter() - started
        self.last_embed_stats = {
            "model": self.model, "total": len(items),
            "cache_hits": hits, "cache_misses": misses,
            "http_batches": n_batches, "http_texts": http_texts_done,
            "elapsed": elapsed, "cache_backend": cache_label, "ok": True,
        }
        # ---- 一条汇总，直接回答「这次到底有没有走 Ollama」----
        if http_texts_done == 0:
            logger.info(
                "嵌入完成：共 %d 条%s = 缓存命中 %d + 未命中 0 —— **本次未调用 Ollama**；"
                "模型=%s 缓存=%s 耗时=%.1fms；embed_seconds 只统计真实 HTTP 调用，"
                "全命中时样本数为 0 属设计（非指标丢失）",
                len(items), dedup_note, hits, self.model, cache_label, elapsed * 1000.0)
        else:
            logger.info(
                "嵌入完成：共 %d 条%s = 缓存命中 %d + 未命中 %d（**真正走 Ollama 的条数**），"
                "HTTP %d 批；模型=%s 缓存=%s 耗时=%.1fms；embed_seconds 样本数应为 %d"
                "（仅未命中项，命中项不产生样本，属设计）",
                len(items), dedup_note, hits, misses, n_batches, self.model, cache_label,
                elapsed * 1000.0, n_batches)
        return result

    # P0.5：查询向量化在首查时要等 bge-m3 冷加载（真机实测 7.6~9.3s），
    # 阈值 500ms —— 命中缓存时只有零点几毫秒，冷加载时一眼可见。
    @timed(operation="embed.embed_query", slow_ms=500.0)
    def embed_query(self, text: str) -> list[float]:
        """单条查询向量（同样走缓存）。"""
        vectors = self.embed_texts([text])
        if not vectors:
            raise RuntimeError(f"嵌入查询失败：Ollama 未返回向量（model={self.model}）")
        return vectors[0]

    # ------------------------------------------------------------------
    # 运维
    # ------------------------------------------------------------------
    def close(self) -> None:
        # 关闭也留痕（含本轮累计）：便于对齐「什么时候不再有新请求」。
        logger.info(
            "嵌入后端关闭：model=%s；本轮 HTTP 调用=%d 次，嵌入文本=%d 条，"
            "缓存命中=%d 次，缓存=%s",
            self.model, self.calls, self.texts_embedded, self.cache_hits,
            self._cache_label())
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            logger.debug("关闭 Ollama 客户端失败", exc_info=True)
        if self.cache is not None:
            self.cache.close()

    def health(self, *, deep: bool = False) -> dict:
        """健康信息；``deep=True`` 时真发一次小请求并上报维度。"""
        info: dict = {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "batch_size": self.batch_size,
            "dim": self._dim,
            "calls": self.calls,
            "texts_embedded": self.texts_embedded,
            "cache": self.cache.stats() if self.cache is not None else {"backend": "none"},
        }
        try:
            tags = self.client.get_json("/api/tags", timeout=5.0, operation="ollama.tags")
            installed = [m.get("name") for m in tags.get("models", [])]
            info["ping"] = True
            info["model_installed"] = any(
                name == self.model or str(name).split(":")[0] == self.model.split(":")[0]
                for name in installed if name)
            info["installed_models"] = installed[:20]
        except Exception as exc:  # noqa: BLE001 - 健康检查失败不抛
            info["ping"] = False
            info["error"] = f"{type(exc).__name__}: {exc}"
        if deep and info.get("ping", False):
            try:
                info["dim"] = self.ensure_dim()
            except Exception as exc:  # noqa: BLE001
                info["dim_error"] = f"{type(exc).__name__}: {exc}"
        return info


def probe_dimension(config: Any, *, texts: Sequence[str] | None = None) -> int:
    """便捷函数：构造一次嵌入后端并实测维度，用完即关（脚本/测试用）。"""
    embedder = OllamaEmbedder(config=config)
    try:
        if texts:
            embedder.embed_texts(list(texts))
            return embedder.dim
        return embedder.ensure_dim()
    finally:
        embedder.close()
