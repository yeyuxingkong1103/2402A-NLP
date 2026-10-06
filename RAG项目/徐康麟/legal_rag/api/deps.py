# -*- coding: utf-8 -*-
"""应用装配：把引擎、记忆、业务数据、长期记忆组装成一个 AppState。

延迟装配（lazy）的原因：让 /health 在依赖缺席（Redis/MySQL/模型）时也能回答，
而不是整个服务起不来。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..business.store import BusinessStore, build_business_store
from ..config import KNOWLEDGE_DIR, RagConfig
from ..engine import RagEngine
from ..ingest.pipeline import KnowledgeBasePipeline, group_sources_by_role
from ..memory.longterm import DEFAULT_MEMORY_COLLECTION, build_longterm_memory
from ..memory.session import SessionStore, build_session_store
from ..roles import DEFAULT_ROLE_ID, ROLE_LIBRARY
from .concurrency import (
    RequestLimiter,
    SessionLockRegistry,
    make_thread_limiter,
)
from .auth import AuthService
from .documents import DocumentService, JobRegistry

logger = logging.getLogger(__name__)

#: ``/health`` 顶层 ``status`` 的分级语义（从轻到重）。
#:
#: * :data:`HEALTH_STATUS_OK`       —— 所有**已装配**子系统都健康；**只有这种情况才是 ok**；
#: * :data:`HEALTH_STATUS_STARTING` —— 引擎尚未装配（lazy 装配：先能回答 /health 再补依赖）；
#: * :data:`HEALTH_STATUS_DEGRADED` —— 仍在服务，但有子系统降级（向量库回落内存库、
#:   Redis/MySQL 不可用回落内存/SQLite 等）；
#: * :data:`HEALTH_STATUS_ERROR`    —— 后端彻底不可达，或子系统 ``health()`` 自身抛异常。
HEALTH_STATUS_OK = "ok"
HEALTH_STATUS_STARTING = "starting"
HEALTH_STATUS_DEGRADED = "degraded"
HEALTH_STATUS_ERROR = "error"

#: 严重度排序：聚合时取最严重的那个。
_HEALTH_SEVERITY: dict[str, int] = {
    HEALTH_STATUS_OK: 0,
    HEALTH_STATUS_STARTING: 1,
    HEALTH_STATUS_DEGRADED: 2,
    HEALTH_STATUS_ERROR: 3,
}

#: 多来源降级原因的连接符（稳定可解析：split 即可还原每个来源）。
REASON_SEP = " | "

#: ``/health`` 顶层「库内**活条数**」字段名。
#:
#: ⚠️ **只允许来自** ``VectorStore.count()``（等价于 Milvus ``query(count(*))``）。
#: **绝对禁止**用 ``num_entities`` / ``get_collection_stats()['row_count']`` 充当对外数值：
#: 它们包含 ``upsert``（= delete + insert）留下的**软删除墓碑**，在 compaction 之前会
#: 明显虚报（实测虚报过 5 倍：row_count=78 而活行只有 16）。
#: 数字看着正常但它是错的 —— 与「降级了却报 ok」属同一类缺陷。
VECTOR_STORE_ROWS_KEY = "vector_store_rows"
#: 活条数的来源标记，写进响应里，防止后人换成会虚报的 API。
VECTOR_STORE_ROWS_SOURCE = "store.count()"


@dataclass
class AppState:
    config: RagConfig
    engine: RagEngine | None = None
    sessions: SessionStore | None = None
    business: BusinessStore | None = None
    longterm: LongTermMemory | None = None
    sources: list[str] = field(default_factory=list)

    # t4 服务层：文档热插拔 / 并发闸门 / 会话锁 / 系统指标采样器
    pipeline: KnowledgeBasePipeline | None = None
    documents: DocumentService | None = None
    jobs: JobRegistry | None = None
    limiter: RequestLimiter | None = None
    session_locks: SessionLockRegistry | None = None
    #: 账号体系（注册/登录/令牌）：与业务数据层**共用同一个 store**，不新建第二套存储
    auth: Any = None
    sampler: Any = None
    started_at: float = field(default_factory=time.time)
    _thread_limiter: Any = None
    _thread_limiter_loop: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: /health 快照缓存（D4 方案 A）：由周期采样器每轮刷新，默认读它
    _health_cache: dict | None = field(default=None, repr=False)
    _health_cache_ts: float | None = field(default=None, repr=False)
    _health_cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _health_sampler: Any = field(default=None, repr=False)

    # ---------- 装配 ----------
    def build(self) -> "AppState":
        self.started_at = time.time()
        self.config.ensure_dirs()
        self.engine = RagEngine(self.config)
        self.sessions = build_session_store(self.config)
        self.business = build_business_store(self.config)
        # 长期记忆：接 Milvus 独立 collection（legal_rag_memory），默认关闭
        self.longterm = build_longterm_memory(
            self.config, self.engine.embedder, self.engine.store,
            collection=DEFAULT_MEMORY_COLLECTION)
        # 上传/入库服务：复用同一个 pipeline 与 store（连接复用，不每请求新建）
        self.pipeline = KnowledgeBasePipeline(self.engine.embedder, self.engine.store,
                                              self.config)
        self.jobs = JobRegistry()
        self.documents = DocumentService(self.pipeline, self.engine.store, self.config,
                                         jobs=self.jobs)
        # 并发闸门与会话锁（进程级单例，生命周期内复用）
        self.limiter = RequestLimiter(self.config.concurrency.max_concurrent_requests)
        self.session_locks = SessionLockRegistry()
        self.register_roles()
        # 账号体系：复用同一个 business store（账号表与业务表同库，不新建第二套存储）
        self.auth = AuthService(self.business, self.config)
        self.auth.ensure()
        if self.auth.require_auth:
            logger.info("强制鉴权已开启（AUTH_REQUIRED/REQUIRE_AUTH=true）："
                        "/chat、/sessions、/documents 将校验登录态")
        else:
            logger.warning("强制鉴权**未开启**（默认）：接口接受任意 user_id —— "
                           "本机调试可以，**对外发布前必须置 AUTH_REQUIRED=true**")
        logger.info(
            "服务装配完成：store=%s embedder=%s llm=%s 并发上限=%d 线程池=%d "
            "长期记忆=%s(%s)",
            getattr(self.engine.store, "name", "?"), self.engine.embedder.name,
            self.config.llm_provider, self.config.concurrency.max_concurrent_requests,
            self.config.concurrency.thread_pool_size,
            "启用" if self.longterm.enabled else "关闭", self.longterm.collection)
        return self

    def blocking_limiter(self) -> Any:
        """线程池容量上限（大小 = ``config.concurrency.thread_pool_size``）。

        必须在**运行中的事件循环**里创建；换了循环就重建（asyncio 原语绑定 loop）。
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._thread_limiter is None or self._thread_limiter_loop is not loop:
                self._thread_limiter = make_thread_limiter(
                    self.config.concurrency.thread_pool_size)
                self._thread_limiter_loop = loop
                logger.debug("重建线程池容量上限：%d（thread_pool_size）",
                             self.config.concurrency.thread_pool_size)
            return self._thread_limiter

    def register_roles(self) -> None:
        if self.business is None:
            return
        for role in ROLE_LIBRARY.values():
            self.business.upsert_role(role.role_id, role.name, role.domain, role.persona)

    # ---------- 知识库 ----------
    def ensure_index(self, rebuild: bool = False) -> dict:
        if self.engine is None:
            self.build()
        assert self.engine is not None
        chunks = self.engine.store.count()
        if chunks > 0 and not rebuild:
            return {"status": "ready", "chunks": chunks}
        # ⚠️ **降级状态下拒绝"启动自愈式整库重建"**（真机踩到，2026-09-24，代价是一次 OOM）：
        # 现场：上一个 API 进程没退干净 ⇒ 新进程打不开 Milvus 文件（`DataDirLockedError`）
        # ⇒ 按既有策略**静默降级成内存库**（这是设计好的、可接受的降级）⇒ 但紧接着本函数看到
        # `count()==0`，把它当成"首次运行"⇒ **发起 2453 篇 / 11.9 万块的整库重建**：
        #   * 往**临时内存库**里灌（白干，还占内存）；
        #   * 同一张卡上 vLLM 已占 37 GB ⇒ 嵌入直接 `CUDA error: out of memory`。
        # 口径：**库不可用（降级）时不自动重建**，只报可见错误并继续以"空知识库"服务
        # （保住了"降级可见但服务照常"的既有约定，又不会干出破坏性/昂贵的事）。
        # 运维显式要求重建（``POST /ingest {"rebuild": true}``）时不受此限制。
        if not rebuild and getattr(self.engine, "degraded", False):
            reason = getattr(self.engine, "degraded_reason", "") or "未知原因"
            logger.error(
                "知识库**降级**（%s：请求 %s，实际 %s）且当前为空 ⇒ **拒绝自动整库重建**"
                "（否则会把整个语料灌进临时库、并可能因显存不足而失败）。"
                "请先排掉占用 Milvus 文件锁的进程再重启；确需重建请显式调用 POST /ingest。",
                reason, getattr(self.engine, "store_requested", "?"),
                getattr(self.engine, "store_actual", "?"))
            return {"status": "skipped_degraded_store", "chunks": chunks,
                    "degraded_reason": reason,
                    "hint": "检查是否有残留进程占用 Milvus 文件锁；显式重建用 POST /ingest"}
        pipeline = KnowledgeBasePipeline(self.engine.embedder, self.engine.store, self.config)
        # 角色隔离的入库侧：knowledge/<role_id>/ 归属该角色（见 group_sources_by_role）。
        # 绝不能整目录一把入库 —— 那样律师知识会变成「没有归属」，谁都能看见。
        groups = group_sources_by_role(self.sources or [str(KNOWLEDGE_DIR)],
                                       default_role=DEFAULT_ROLE_ID)
        files: list[dict] = []
        totals = {"documents": 0, "skipped": 0, "failed": 0, "chunks": 0}
        elapsed = 0.0
        for role_id, paths in groups:
            logger.info("按角色入库：role_id=%r，%d 个来源", role_id, len(paths))
            report = pipeline.ingest(paths, rebuild=True, role_id=role_id)
            for item in report.files:
                files.append({**item, "role_id": role_id})
            totals["documents"] += report.documents
            totals["skipped"] += report.skipped
            totals["failed"] += report.failed
            totals["chunks"] += report.chunks
            elapsed += report.elapsed
        return {
            "status": "ingested",
            "roles": {role_id: paths for role_id, paths in groups},
            "documents": totals["documents"],
            "skipped": totals["skipped"],
            "failed": totals["failed"],
            "chunks": totals["chunks"],
            "elapsed": round(elapsed, 3),
            "files": files,
        }

    # ---------- 运维 ----------
    @staticmethod
    def _issue(source: str, status: str, reason: str = "") -> dict:
        """一条聚合用的健康问题（``health_issues`` 的元素，结构稳定）。"""
        return {"source": source, "status": status, "reason": str(reason or "")}

    @classmethod
    def _collect_health(cls, source: str, component) -> tuple[object, dict | None]:
        """调用子系统 ``health()``，并把结果翻译成 (payload, issue)。

        * 组件为 ``None`` 或没有 ``health()`` -> ``(None, None)``，不参与聚合；
        * ``health()`` 抛异常 -> payload 置 None 并返回 ``error`` issue
          （**健康检查自己绝不 500**，这正是 /health 存在的意义）；
        * 否则按 payload 的字段判出问题，语义见 :data:`HEALTH_STATUS_DEGRADED`。
        """
        if component is None:
            return None, None
        getter = getattr(component, "health", None)
        if not callable(getter):
            return None, None

        try:
            payload = getter()
        except Exception as exc:  # noqa: BLE001 - 健康检查自身异常不能外溢
            logger.warning("子系统健康检查失败（%s）：%s: %s",
                           source, type(exc).__name__, exc, exc_info=True)
            return None, cls._issue(
                source, HEALTH_STATUS_ERROR,
                f"health() 抛异常：{type(exc).__name__}: {exc}")

        if not isinstance(payload, dict):
            return payload, None

        status = payload.get("status")
        if status == HEALTH_STATUS_ERROR:
            return payload, cls._issue(
                source, HEALTH_STATUS_ERROR,
                payload.get("error") or payload.get("degraded_reason") or "")
        if status == HEALTH_STATUS_DEGRADED or payload.get("degraded") is True:
            return payload, cls._issue(
                source, HEALTH_STATUS_DEGRADED,
                payload.get("degraded_reason") or payload.get("error") or "")
        # 交叉核对：配置想要的后端与实际生效的后端不一致 => 已降级
        # （engine 自己会把 status 置为 degraded，这里是第二道防线，只读其字段）
        requested = payload.get("store_requested")
        actual = payload.get("store_actual")
        if requested and actual and str(requested).strip().lower() != str(actual).strip().lower():
            return payload, cls._issue(
                source, HEALTH_STATUS_DEGRADED,
                f"向量库实际使用 {actual}，与配置要求的 {requested} 不一致")
        # 其余子系统（sessions / business / longterm）：ping 失败或带 error 即视为降级
        if payload.get("ping") is False:
            return payload, cls._issue(
                source, HEALTH_STATUS_DEGRADED,
                payload.get("error") or "ping 失败")
        if payload.get("error"):
            return payload, cls._issue(
                source, HEALTH_STATUS_DEGRADED, str(payload["error"]))
        return payload, None

    def health(self) -> dict:
        """组件健康状态；**顶层 ``status`` 由子系统健康状态聚合得出**。

        分级语义（由轻到重，聚合时取最严重者）
        ---------------------------------------
        ================  ============================================================
        顶层 ``status``    含义
        ================  ============================================================
        ``ok``             所有**已装配**子系统健康 —— 只有这种情况才是 ok
        ``starting``       引擎尚未装配（lazy 装配：依赖缺席时 /health 也要能回答）
        ``degraded``       仍在服务，但有子系统降级（向量库回落内存库、Redis/MySQL
                           回落内存/SQLite、ping 失败……）
        ``error``          后端彻底不可达，或子系统 ``health()`` 本身抛异常
        ================  ============================================================

        ⚠️ **为什么顶层必须聚合**
        t7 阶段出过「服务已降级但 /health 仍报 ok」的 blocker：降级信号只埋在内层
        ``result["engine"]``，而顶层 ``status`` 恒为 ``"ok"``，于是只读顶层字段的
        调用方 / 监控 / 验收脚本照样把服务判为健康。现在 ``status`` 由
        ``engine`` / ``sessions`` / ``business`` / ``longterm`` 聚合得出，
        并由 ``degraded_reason`` + ``health_issues`` 让原因在顶层可直接定位，
        **无需二次解析 ``result["engine"]``**。

        向后兼容
        --------
        ``config`` / ``engine`` / ``sessions`` / ``business`` / ``longterm`` 五个键
        全部保留、不改名不删除、结构不变；``status`` 的**聚合分级语义与 t13 完全一致**
        （``ok``/``starting``/``degraded``/``error`` + ``degraded_reason`` +
        ``health_issues`` + ``ready``），t4 只在**其后追加**这些新键：

        * ``backends``      —— 各后端可达性（up/down + 延迟）与 GPU 可用性（复用 t6 ``health()``）
        * ``gpu``           —— ``backends.gpu`` 的快捷引用（无卡时 ``available=false`` + ``reason``）
        * ``concurrency``   —— 并发上限/线程池/排队数/会话锁统计
        * ``documents``     —— 上传目录、job 数、**库内活条数**（来源标注 ``store.count()``）
        * ``uptime_seconds``—— 服务启动至今

        两条硬规则（"虚假数字"防线）：
        1. 活条数**只能**来自 ``store.count()``（``query(count(*))``），
           **禁止** ``num_entities`` / ``row_count``（含 upsert 墓碑，实测虚报 5 倍）；
        2. 该值是 **Gauge（存量，会随删除下降）**，与 ``docs_total`` / ``chunks_total``
           这类 **Counter（进程内累计写入，rebuild 会涨、重启归零）** 不是一回事。
        """
        result: dict = {
            "config": {
                "vector_store": self.config.vector_store,
                "embedding_provider": self.config.embedding_provider,
                "rerank_provider": self.config.rerank_provider,
                "llm_provider": self.config.llm_provider,
            },
        }

        issues: list[dict] = []
        for source, component in (("engine", self.engine),
                                  ("sessions", self.sessions),
                                  ("business", self.business),
                                  ("longterm", self.longterm)):
            payload, issue = self._collect_health(source, component)
            if payload is not None:
                result[source] = payload
            if issue is not None:
                issues.append(issue)

        engine_missing = self.engine is None
        # 先按严重度定级：任一 error -> error；否则引擎未装配 -> starting；
        # 否则有降级 -> degraded；全健康 -> ok。
        if any(item["status"] == HEALTH_STATUS_ERROR for item in issues):
            status = HEALTH_STATUS_ERROR
        elif engine_missing:
            status = HEALTH_STATUS_STARTING
        elif issues:
            status = HEALTH_STATUS_DEGRADED
        else:
            status = HEALTH_STATUS_OK

        # 顶层降级原因：engine 的原因**原样放在最前**（便于直接用 engine 的报错文本定位），
        # 其余来源以 " | " 追加，稳定可解析（split(REASON_SEP) 即得各来源）。
        reasons: list[str] = []
        engine_issue = next((i for i in issues if i["source"] == "engine"), None)
        if engine_issue and engine_issue["reason"]:
            reasons.append(engine_issue["reason"])
        for item in issues:
            if item is engine_issue or not item["reason"]:
                continue
            reasons.append(f'{item["source"]}: {item["reason"]}')
        if not reasons and engine_missing:
            reasons.append("引擎尚未装配（lazy 装配语义）：/health 可用，但主链路未就绪")

        result["status"] = status
        result["degraded_reason"] = REASON_SEP.join(reasons) if reasons else None
        result["health_issues"] = issues
        result["ready"] = not engine_missing

        # 库内「活条数」：取自 engine.health()['chunks']，而后者就是 ``store.count()``
        # （Milvus 侧为 ``query(count(*))``）—— 运维不查库也能看到存量。
        # ⚠️ 绝不改成 num_entities / row_count（含 upsert 墓碑，会虚报）。
        # count() 失败时 engine.health() 会抛异常，已被上面的 _collect_health 兜成
        # error 状态，这里自然拿到 None，返回可解释状态而不是让 /health 崩。
        engine_payload = result.get("engine")
        rows = engine_payload.get("chunks") if isinstance(engine_payload, dict) else None
        # 库内「活条数」：**单一口径** —— 先让 DocumentService 用 ``store.count()``
        # 刷新一次并发布 Gauge，然后**同一个值**同时用于顶层字段与 documents 块，
        # 绝不出现「顶层 2、documents 块 0」这种两套口径（那正是"虚假数字"的来源）。
        # ⚠️ 绝不改成 num_entities / row_count（含 upsert 墓碑，会虚报）。
        # count() 失败时返回 None + 可解释状态，而不是让 /health 崩。
        live_rows: int | None = None
        if self.documents is not None:
            live_rows = self.documents.store_rows(force=True)
        if live_rows is None:
            engine_payload = result.get("engine")
            raw = engine_payload.get("chunks") if isinstance(engine_payload, dict) else None
            try:
                live_rows = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                live_rows = None
        result[VECTOR_STORE_ROWS_KEY] = live_rows
        result["vector_store_rows_source"] = (
            VECTOR_STORE_ROWS_SOURCE if live_rows is not None else None)

        # ---------- t4 追加：后端可达性 / GPU / 并发 / 文档（**只增不改**上面两层语义）----------
        result["backends"] = self.backend_health()
        result["gpu"] = result["backends"].get("gpu")
        result["concurrency"] = self.concurrency_snapshot()
        result["documents"] = self.documents_snapshot(rows=live_rows)
        result["uptime_seconds"] = round(time.time() - self.started_at, 3)
        return result

    # ---------- /health 快照缓存（D4 方案 A，2026-09-28 拍板）----------
    #
    # 为什么必须缓存：`health()` 每次都**现采**（依赖探针 + `store_rows(force=True)` 查库），
    # 依赖不可达时单次 = 探针超时（实测 Redis 挂掉时 **2.2 s**），且占一个工作线程
    # ⇒ 8 线程下吞吐被钉在 ~3.6 QPS（见 docs/LOAD-TEST.md §3）。
    # 现在默认读快照（O(1)、**不占线程池**），要现采就显式 `/health?deep=1`。
    #
    # ⚠️ 关键约束（验收第 3 条）：采样器停了（`SYS_METRICS_ENABLED=0`）就**没有新快照**，
    #    此时**必须回落现采**并在响应里标 `stale=true` / `snapshot_age_seconds` ——
    #    **不许把旧数据当新鲜数据发出去**。
    def attach_health_sampler(self, sampler: object) -> None:
        """把周期采样器记下来：既靠它刷新快照，也靠 `is_running()` 判断快照还新不新。"""
        self._health_sampler = sampler

    def refresh_health_cache(self) -> dict | None:
        """采一次**完整** health 存成快照（由周期采样器的回调每轮调用）。

        采集失败**保留旧快照**并告警（不把缓存清空，否则运维会看到"突然没有健康信息"）。
        """
        try:
            payload = self.health()
        except Exception as exc:  # noqa: BLE001 - 快照刷新绝不能把采样线程带崩
            logger.warning("[快照] health 采集失败（保留旧快照）：%s", exc, exc_info=True)
            return None
        with self._health_cache_lock:
            self._health_cache = payload
            self._health_cache_ts = time.time()
        return payload

    def _cached_health(self) -> tuple[dict, float] | None:
        with self._health_cache_lock:
            payload = self._health_cache
            ts = self._health_cache_ts
        if payload is None or ts is None:
            return None
        return payload, max(0.0, time.time() - ts)

    def snapshot_age_seconds(self) -> float | None:
        cached = self._cached_health()
        return None if cached is None else round(cached[1], 1)

    def health_payload(self, *, deep: bool = False) -> dict:
        """`/health` 的响应体：默认读**快照**，`deep=True` 现采。

        追加三个键（**只增不改**，向后兼容）：
        * ``check``               —— ``"snapshot"`` 或 ``"live"``（这次是怎么来的）
        * ``snapshot_age_seconds``—— 快照距现在多少秒（现采且从未采过时为 ``None``）
        * ``stale``               —— 是否**没拿到新鲜快照**（此时内容是现采的）
        """
        sampler = self._health_sampler
        sampler_running = bool(sampler is not None and getattr(sampler, "is_running", None)
                               and sampler.is_running())
        cached = self._cached_health()
        age = None if cached is None else round(cached[1], 1)
        stale_after = self._health_stale_after()

        if not deep and sampler_running and cached is not None:
            payload = dict(cached[0])
            fresh = cached[1] <= stale_after
            payload["check"] = "snapshot"
            payload["snapshot_age_seconds"] = age
            payload["stale"] = not fresh
            if not fresh:
                payload["snapshot_note"] = (
                    f"快照已 {age}s 未更新（超过 {stale_after:.0f}s）：采样线程可能卡住或间隔被调大")
                logger.warning("[快照] /health 读到的快照偏旧：age=%.1fs > %.0fs",
                               cached[1], stale_after)
            self._count_health("snapshot")
            return payload

        # 现采：`deep=1`，或**没有可用的新鲜快照**（采样器没在跑 / 还没采过第一轮）
        payload = self.health()
        payload["check"] = "live"
        payload["snapshot_age_seconds"] = age
        payload["stale"] = not deep
        if not deep:
            reason = ("系统指标采样未启用或已停止（SYS_METRICS_ENABLED=0）"
                      if not sampler_running else "尚无快照（采样器刚启动、还没跑完第一轮）")
            payload["snapshot_note"] = f"{reason} ⇒ 本次为现采，代价是探针耗时"
            logger.info("[快照] 无可用的新鲜快照（%s）-> /health 回落现采", reason)
        self._count_health("live")
        return payload

    def _health_stale_after(self) -> float:
        """快照超过多久算"不新鲜"。默认 2 倍采样间隔（至少 30 s），可由采样器间隔推导。"""
        interval = 15.0
        sampler = self._health_sampler
        try:
            interval = float(getattr(sampler, "interval", interval))
        except (TypeError, ValueError):
            pass
        return max(30.0, interval * 2)

    @staticmethod
    def _count_health(mode: str) -> None:
        try:
            from .. import metrics as M  # noqa: PLC0415 - 延迟导入，避免模块级循环

            M.counter("health_check_total").inc(mode=mode)
        except Exception:  # noqa: BLE001 - 指标失败不该影响健康检查
            logger.debug("health_check_total 计数失败（忽略）", exc_info=True)

    # ---------- t4 运维视图（都不抛异常）----------
    def backend_health(self, *, timeout: float = 5.0) -> dict:
        """各后端可达性（up/down + 延迟）与 GPU 可用性 —— 直接复用 t6 的 ``health()``。

        t6 已经做了：``checks{redis,milvus,ollama}.{up,latency_ms,reason}``、
        ``gpu{available,reason,memory_*}``、``ollama{loaded_count,models}``。
        这里只负责「拿过来 + 永不抛」。
        """
        try:
            from .. import system_metrics as sysm

            payload = sysm.health(config=self.config, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - 健康检查自身绝不外溢
            logger.warning("系统健康检查失败：%s: %s", type(exc).__name__, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "checks": {},
                    "gpu": {"available": False, "reason": "系统指标不可用"}}
        checks = payload.get("checks") or {}
        summary = {
            "ok": bool(payload.get("ok")),
            "checks": checks,
            "down": sorted(name for name, item in checks.items()
                           if isinstance(item, dict) and not item.get("up")),
            "gpu": payload.get("gpu") or {"available": False},
            "ollama": payload.get("ollama"),
            "elapsed_ms": payload.get("elapsed_ms"),
            "errors": payload.get("errors") or [],
        }
        return summary

    def concurrency_snapshot(self) -> dict:
        return {
            "max_concurrent_requests": self.config.concurrency.max_concurrent_requests,
            "thread_pool_size": self.config.concurrency.thread_pool_size,
            "limiter": self.limiter.snapshot() if self.limiter else None,
            "session_locks": self.session_locks.snapshot() if self.session_locks else None,
        }

    def documents_snapshot(self, *, rows: int | None = None) -> dict:
        if self.documents is None:
            return {"ready": False}
        payload = {
            "ready": True,
            "upload_dir": str(self.documents.uploads.directory),
            "jobs_tracked": len(self.jobs) if self.jobs else 0,
            "rows": self.documents.rows_snapshot(),
        }
        if rows is not None:
            # 与顶层 vector_store_rows **同一个值**（单一口径）
            payload["rows"]["value"] = rows
        return payload

    def start_sampler(self) -> Any:
        """启动 t6 的周期采样器（把 process_*/host_*/gpu_*/dep_* 持续发布到指标注册表）。

        采样回调里顺带刷新「库内活条数」Gauge —— 这样 ``GET /metrics`` 只渲染注册表，
        **不需要**自己去做任何后端调用（保持轻量）。
        """
        try:
            from .. import system_metrics as sysm

            interval = float(self.config.logging.sys_metrics_interval)

            def _on_sample(_snap: dict) -> None:
                if self.documents is not None:
                    self.documents.store_rows(force=True)

            if not getattr(self.config.logging, "sys_metrics_enabled", True):
                logger.info("系统指标采样器未启用（SYS_METRICS_ENABLED=false）")
                if self.documents is not None:
                    self.documents.store_rows(force=True)   # 至少把存量发布一次
                # ⚠️ 采样器关着 ⇒ 没有新鲜快照 ⇒ /health 会**回落现采**并标 stale=true
                #    （D4 验收第 3 条：不许把旧数据当新鲜数据发出去）
                logger.info("/health 将回落为现采（无快照可用）")
                return None
            self.sampler = sysm.install_periodic_sampler(interval, config=self.config,
                                                         on_sample=_on_sample)
            # D4 方案 A：让采样器**每轮顺带刷新完整 health 快照**（/health 默认读它）。
            # 用 add_on_sample 而不是构造参数：install_periodic_sampler 在实例已运行时
            # 会直接复用，构造参数里的回调就传不进去了。
            self.attach_health_sampler(self.sampler)
            if getattr(self.sampler, "add_on_sample", None):
                self.sampler.add_on_sample(lambda _snap: self.refresh_health_cache())
            else:  # pragma: no cover - 只可能在降级/替身实现时发生
                logger.warning("采样器不支持追加回调 -> /health 只能现采（功能不受影响，只是变慢）")
            logger.info("系统指标采样器已启动：每 %.1fs 采样一次（并刷新 /health 快照）", interval)
        except Exception as exc:  # noqa: BLE001 - 采样器失败不影响服务
            logger.warning("系统指标采样器启动失败（不影响服务）：%s: %s",
                           type(exc).__name__, exc)
        return self.sampler

    def stop_sampler(self) -> None:
        sampler = self.sampler
        if sampler is None:
            return
        try:
            sampler.stop()
            logger.info("系统指标采样器已停止")
        except Exception as exc:  # noqa: BLE001
            logger.warning("停止系统指标采样器失败：%s: %s", type(exc).__name__, exc)
        finally:
            self.sampler = None

    def close(self) -> None:
        self.stop_sampler()
        if self.business is not None:
            self.business.close()
