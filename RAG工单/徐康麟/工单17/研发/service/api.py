# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

容器对外 FastAPI 服务（**工单17 优化后**）：``研发/service/api.py``。

启动（容器内）::

    uvicorn service.api:app --host 0.0.0.0 --port 8200

启动（本机等价验证，与容器同一份代码）::

    python -m uvicorn service.api:app --host 127.0.0.1 --port 8200

HTTP 契约（详见 ``设计/接口设计.md``）：

============================  ======  ==================================================
路径                           方法    说明
============================  ======  ==================================================
``/``                         GET     服务信息（版本/端口/后端/会话统计）
``/health``                   GET     健康检查：index / llm(ollama) / redis / 卷可写性
``/healthz``                  GET     存活探针（不碰索引，容器 HEALTHCHECK 用）
``/readyz``                   GET     就绪探针（索引必须可用，否则 503）
``/metrics``                  GET     Prometheus 文本格式（**工单17 新增限流/连接池/缓存/队列指标**）
``/api/v1/ask``               POST    金融问答（核心接口，**工单17：异步路由 + 入口限流**）
``/api/v1/sessions``          GET     卷/Redis 里的会话列表（容器间共享的可观测证据）
``/api/v1/sessions/{id}``     GET     指定会话的多轮历史
``/api/v1/index``             GET     索引元信息（块数/维度/模态分布）
``/api/v1/retrieve``          POST    只检索不生成（诊断）
**``/api/v1/uploads``**       POST    **新增**：上传 PDF → 解析任务（优化后立即 202）
**``/api/v1/tasks``**         GET     **新增**：解析任务列表与队列深度
**``/api/v1/tasks/{id}``**    GET     **新增**：解析任务状态
**``/api/v1/admin/unload``**  POST    **新增**：主动卸载 Ollama 模型（仅回环）
**``/api/v1/perf``**          GET     **新增**：性能自观测快照
``/openapi.json``             GET     自动生成 OpenAPI 3.1（FastAPI 自带）
============================  ======  ==================================================

工单17 的十项修复（F1~F10，全部受 ``PERF_ENABLED`` 总开关控制；关闭时行为与工单10 基座
**逐位一致**）见 ``设计/优化方案设计.md`` §1。本文件承载其中的：

* **F3 入口限流**：``service/limiter.py`` 的有界闸门，超出即 ``429``（不再堆积）；
* **F4 异步路由**：``ask`` 改 ``async def`` + ``run_in_threadpool``，
  事件循环不被阻塞（``/healthz`` 在高压下仍可答）；
* **F9 解析队列**：``POST /api/v1/uploads`` → 有界队列 + 工作线程池（对应 RAGFlow 的 Task Executor）；
  ``PERF_ENABLED=0`` 时**同步解析**（如实重建原文描述的「API Server 被解析阻塞」反模式，
  用于产出可对比的基准数据）；
* **F10 健康检查缓存**：``/health`` 结果 TTL 缓存，探针不再反复打「载索引 + 读盘」的重活。

设计纪律（继承工单10）：
    * 业务逻辑**零复制**：全部转发到 ``app.core.qa_engine.QAEngine``；
    * 会话落盘 + Redis 双写，见 :mod:`service.session_store`；
    * 每个请求一行 JSON 日志（``DEPLOY_LOG_JSON=1``）；
    * 健康检查**不伪装**：未就绪项如实给 ``ok=false`` 与原因字符串；
    * 所有阈值来自环境变量（:mod:`service.config` / :mod:`service.perf_config`），代码里不写死。
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# —— 让容器内 ``uvicorn service.api:app`` 能 import 到 app 包（/app/service → /app/app）——
if __package__ in (None, ""):   # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import describe_runtime, get_config          # noqa: E402
from app.core.errors import IndexMissingError, LLMProbeError, RagError   # noqa: E402
from app.core.llm_client import unload_models                      # noqa: E402
from app.core.logging_conf import get_logger, setup_logging        # noqa: E402
from app.core.qa_engine import QAEngine                            # noqa: E402
from app.storage.sqlite_manager import SqliteManager               # noqa: E402

from .config import DeployConfig, get_deploy_config                # noqa: E402
from .http_pool import get_http_pool                               # noqa: E402
from .limiter import GateRejected, GateTimeout, get_gate           # noqa: E402
from .perf_config import PerfConfig, get_perf_config               # noqa: E402
from .session_store import SharedSessionStore, Turn                # noqa: E402
from .singletons import registry_stats                             # noqa: E402
from .task_queue import UploadRejected, get_parse_queue, parse_sync  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"
API_VERSION = "1.7.0"

#: ★ 专用于「等闸门」的独立线程池（**绝不能复用 anyio 的线程池**）
#:
#: 实测踩坑（由 `测试/测试数据/限流验证.json` 抓出，两个连环 bug）：
#:   ① 最早版本直接在事件循环上 `gate.acquire()` —— 排队时阻塞整个循环，
#:      而持票方的 `await run_in_threadpool(_ask_sync)` 完成后要靠**同一个循环**分发回调
#:      才能执行 `finally: gate.release()` ⇒ 票据永不归还 ⇒ **服务假死**；
#:   ② 改成 `await run_in_threadpool(gate.acquire)` 后，等待者与 `_ask_sync` **抢同一个
#:      anyio 线程池**（默认 40 线程）：等待者把线程占满 ⇒ `_ask_sync` 拿不到线程 ⇒
#:      槽位不释放 ⇒ 等到 acquire 超时为止，持续高压下反复死锁。
# 正解：等待动作放到**独立的、有界的**线程池里，与业务执行的线程池彻底隔离。
# 池大小 = 闸门的排队上限（至少 4、至多 64），因此「等闸门」的并发数永远不会
# 超过 ``PERF_QUEUE_MAX`` —— 有界限流的语义不被线程池悄悄放大。

_STARTED_AT = time.time()
_METRICS: dict[str, float] = {
    "rag_api_requests_total": 0.0,
    "rag_api_errors_total": 0.0,
    "rag_api_ask_total": 0.0,
    "rag_api_refused_total": 0.0,
    "rag_api_ask_latency_ms_sum": 0.0,
    "rag_api_ask_latency_ms_max": 0.0,
    # —— 工单17 新增 ——
    "rag_api_rejected_total": 0.0,
    "rag_api_gate_timeout_total": 0.0,
    "rag_api_uploads_total": 0.0,
    "rag_api_upload_rejected_total": 0.0,
    "rag_api_ask_latency_ms_sum_sq": 0.0,
}
_METRICS_LOCK = threading.Lock()


# ====================================================================== #
# 请求/响应模型
# ====================================================================== #
class AskRequest(BaseModel):
    """``POST /api/v1/ask`` 请求体（工单10 契约，逐位不变）。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题（中文/英文）")
    session_id: str | None = Field(None, max_length=128, description="会话 id；多轮追问必传同一 id")
    lang: str = Field("auto", pattern="^(auto|zh|en)$", description="语言路径")
    top_k: int | None = Field(None, ge=1, le=20, description="检索条数（默认取 RAG_RETRIEVAL__TOP_K）")
    with_evidence: bool = Field(False, description="是否返回证据块（诊断用，响应体更大）")


class AskResponse(BaseModel):
    """``POST /api/v1/ask`` 响应体（字段与工单7 的 ``Answer`` 对齐；工单10 契约不变）。"""

    question: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    cited_pages: list[int] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    backend: str = ""
    model: str = ""
    generation_method: str = ""
    language: str = "zh"
    trace_id: str = ""
    session_id: str | None = None
    cache: dict[str, Any] = Field(default_factory=dict)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    session_turns: int = 0
    session_backend: str = ""


# ====================================================================== #
# 运行时单例
# ====================================================================== #
class Runtime:
    """进程级运行时：QAEngine + SharedSessionStore + 限流闸门 + 解析队列 + 日志句柄。"""

    def __init__(self, cfg: DeployConfig, perf: PerfConfig) -> None:
        self.deploy = cfg
        self.perf = perf
        self.engine: QAEngine | None = None
        self.sessions: SharedSessionStore | None = None
        self.log: Any = None
        self.error: str = ""
        self.warmup: dict[str, Any] = {}
        self.init_ms: float = 0.0
        self.runtime_db: str = ""
        self.gate = get_gate(perf)
        #: 「等闸门」专用线程池（与 anyio 业务线程池隔离；大小 = 排队上限）
        self.gate_wait_pool = ThreadPoolExecutor(
            max_workers=max(4, min(int(self.gate.queue_max) or 4, 64)),
            thread_name_prefix="gate-wait")
        self.parse_queue: Any = None
        self._ready = threading.Event()
        self._health_cache: tuple[float, dict[str, Any], int] | None = None
        self._health_lock = threading.Lock()
        self.unload_count = 0
        self.last_unload: dict[str, Any] | None = None

    # -- 初始化 -------------------------------------------------------- #
    def setup(self) -> None:
        started = time.perf_counter()
        Path(self.deploy.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.deploy.session_dir).mkdir(parents=True, exist_ok=True)
        Path(self.deploy.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.deploy.runtime_dir).mkdir(parents=True, exist_ok=True)

        app_cfg = get_config(refresh=True)
        # 日志：业务日志与容器日志同目录，容器里挂到日志卷上（/app/logs）
        os.environ.setdefault("RAG_LOG__DIR", self.deploy.log_dir)
        try:
            self.log = setup_logging(app_cfg, run_id=f"container-{int(_STARTED_AT)}")
        except Exception:  # noqa: BLE001 —— 日志初始化失败不阻断服务
            self.log = get_logger("service.api")

        self.sessions = SharedSessionStore(self.deploy)

        # 工单17 F9：解析队列（优化后才有；关闭时上传走同步解析）
        if self.perf.enabled:
            try:
                self.parse_queue = get_parse_queue(
                    self.perf, upload_dir=Path(self.deploy.upload_dir),
                    log_dir=Path(self.deploy.log_dir), logger=self.log)
            except Exception as exc:  # noqa: BLE001 —— 队列起不来不能让服务起不来
                self._warn("api.parse_queue_failed", error=f"{type(exc).__name__}: {exc}")
                self.parse_queue = None

        # 运行期 sqlite（问答历史/答案缓存）必须落到**可写卷**：
        # 索引卷是 `:ro` 只读挂载，索引目录里写的 qa_runtime.sqlite3 在容器里必然失败，
        # 所以优先用 DEPLOY_RUNTIME_DIR，并做一次真实写探针确认。
        db_dir = Path(self.deploy.runtime_dir)
        if not _writable(db_dir):
            fallback = Path(app_cfg.paths.index_dir)
            self._warn("api.runtime_dir_not_writable",
                       path=str(db_dir), fallback=str(fallback))
            db_dir = fallback
        db_path = db_dir / "qa_runtime.sqlite3"
        self.runtime_db = str(db_path)
        storage = None
        try:
            storage = SqliteManager(db_path, logger=self.log)
        except Exception as exc:  # noqa: BLE001 —— 持久化失败不阻断问答（显式降级）
            self._warn("api.storage_failed", path=str(db_path),
                       error=f"{type(exc).__name__}: {exc}")

        self.engine = QAEngine(cfg=app_cfg, logger=self.log, storage=storage)
        if self.deploy.warmup:
            try:
                self.warmup = self.engine.warmup(with_generate=self.deploy.warmup_with_generate)
            except Exception as exc:  # noqa: BLE001 —— 预热失败照常提供服务（首个请求变慢）
                self.error = f"预热失败（服务继续，首请求将变慢）：{type(exc).__name__}: {exc}"
                self._warn("api.warmup_failed", error=self.error)
        self.init_ms = round((time.perf_counter() - started) * 1000, 2)
        self._ready.set()
        self._info("api.ready", init_ms=self.init_ms, port=self.deploy.api_port,
                   pid=os.getpid(), runtime=describe_runtime(),
                   runtime_db=self.runtime_db, session_backend=self.sessions.backend,
                   perf_enabled=self.perf.enabled,
                   max_inflight=self.gate.max_inflight, queue_max=self.gate.queue_max)

    def _warn(self, event: str, **fields: Any) -> None:
        try:
            if self.log is not None:
                self.log.log_event(event, level="WARNING", **fields)
        except Exception:  # noqa: BLE001
            pass

    def _info(self, event: str, **fields: Any) -> None:
        try:
            if self.log is not None:
                self.log.log_event(event, **fields)
        except Exception:  # noqa: BLE001
            pass

    # -- 健康检查缓存（F10） -------------------------------------------- #
    def health_cached(self) -> tuple[dict[str, Any], int]:
        """``/health`` 结果 TTL 缓存：探针不再每次打「载索引 + 读盘」的重活。"""
        ttl = float(self.perf.health_cache_s) if self.perf.enabled else 0.0
        if ttl <= 0:
            return _health_payload_uncached()
        now = time.time()
        with self._health_lock:
            cached = self._health_cache
            if cached is not None and (now - cached[0]) < ttl:
                body = dict(cached[1])
                body["cached"] = True
                body["cache_age_s"] = round(now - cached[0], 2)
                return body, cached[2]
        body, code = _health_payload_uncached()
        with self._health_lock:
            self._health_cache = (time.time(), body, code)
        body = dict(body)
        body["cached"] = False
        return body, code

    # -- 查询 ---------------------------------------------------------- #
    @property
    def ready(self) -> bool:
        return self.engine is not None and self._ready.is_set()


RUNTIME = Runtime(get_deploy_config(), get_perf_config())


# ====================================================================== #
# 应用生命周期
# ====================================================================== #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化引擎（同步阻塞，容器启动期一次性完成）；退出时关闭连接。"""
    RUNTIME.setup()
    try:
        yield
    finally:
        try:
            if RUNTIME.engine is not None and getattr(RUNTIME.engine, "storage", None) is not None:
                RUNTIME.engine.storage.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if RUNTIME.parse_queue is not None:
                RUNTIME.parse_queue.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            get_http_pool().reset()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(
    title=RUNTIME.deploy.api_title,
    version=API_VERSION,
    description=f"工单编号：{WORK_ORDER} —— 并发瓶颈与资源泄漏修复后的对外 API（RAG 基座：工单7/10）",
    lifespan=lifespan,
)


# ====================================================================== #
# 中间件：请求留痕（结构化日志 + 指标 + 响应头）
# ====================================================================== #
@app.middleware("http")
async def observe(request: Request, call_next):
    """记录每个请求：request_id / 路径 / 状态码 / 耗时；写 JSON 日志并累加指标。"""
    trace = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    started = time.perf_counter()
    status = 500
    response = None
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        with _METRICS_LOCK:
            _METRICS["rag_api_requests_total"] += 1
            if status >= 400:
                _METRICS["rag_api_errors_total"] += 1
        try:
            if RUNTIME.log is not None and not request.url.path.startswith(("/healthz", "/metrics")):
                RUNTIME.log.log_event(
                    "http.request", trace_id=trace, method=request.method,
                    path=request.url.path, query=str(request.url.query)[:200],
                    status=status, elapsed_ms=elapsed_ms,
                )
        except Exception:  # noqa: BLE001 —— 日志失败绝不影响响应
            pass
        try:
            if response is not None:
                response.headers["x-request-id"] = trace
                response.headers["x-process-time-ms"] = str(elapsed_ms)
        except Exception:  # noqa: BLE001
            pass


# ====================================================================== #
# 探针与信息
# ====================================================================== #
@app.get("/healthz", summary="存活探针（不碰索引）")
def healthz() -> dict[str, Any]:
    """仅表示进程存活 —— 容器 HEALTHCHECK 用它，避免索引慢导致容器被误判不健康。"""
    return {"status": "alive", "pid": os.getpid(), "uptime_s": round(time.time() - _STARTED_AT, 2)}


@app.get("/readyz", summary="就绪探针（索引必须可用）")
def readyz() -> Response:
    """索引不可用时返回 503（编排系统据此摘流量）。"""
    body, code = RUNTIME.health_cached()
    return JSONResponse(status_code=(200 if code == 200 else 503), content=body)


@app.get("/health", summary="健康检查：索引 / LLM / Redis / 卷")
def health() -> JSONResponse:
    """健康检查。始终 200（诚实报告每一子项的 ok），便于运维一眼看清降级原因。

    工单17 F10：结果按 ``PERF_HEALTH_CACHE_S``（默认 5 s）缓存 ——
    否则监控每 5 s 探一次就会触发一次「载索引 + 读盘」的重活。
    """
    body, _ = RUNTIME.health_cached()
    return JSONResponse(status_code=200, content=body)


def _health_payload_uncached() -> tuple[dict[str, Any], int]:
    """汇总健康检查：容器层（卷/Redis/端口）+ 业务层（QAEngine.health）。"""
    cfg = RUNTIME.deploy
    checks: dict[str, Any] = {}

    # ① 卷可写性（数据持久化验收的运行期证据）
    checks["volume_logs"] = _dir_writable(cfg.log_dir, "日志卷")
    checks["volume_sessions"] = _dir_writable(cfg.session_dir, "会话卷")
    checks["volume_index"] = _dir_readable(cfg.index_dir, "索引卷")
    checks["volume_uploads"] = _dir_writable(cfg.upload_dir, "上传卷")
    checks["volume_runtime"] = _dir_writable(cfg.runtime_dir, "运行期卷（sqlite/缓存）")

    # ② 会话共享后端
    if RUNTIME.sessions is not None:
        checks["session_store"] = {"ok": True, **RUNTIME.sessions.health()}
    else:
        checks["session_store"] = {"ok": False, "reason": "会话存储未初始化"}

    # ③ 业务引擎（索引 / Ollama / FAISS 等）
    if RUNTIME.engine is None:
        checks["engine"] = {"ok": False, "reason": RUNTIME.error or "引擎未初始化"}
    else:
        try:
            core = RUNTIME.engine.health()
            checks["engine"] = {"ok": core.get("status") == "ok", **core}
        except Exception as exc:  # noqa: BLE001
            checks["engine"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    ready = bool(checks["engine"].get("ok"))
    body = {
        "status": "ok" if (ready and all(c.get("ok") for c in checks.values())) else "degraded",
        "ready": ready,
        "work_order": WORK_ORDER,
        "api_version": API_VERSION,
        "build_ref": cfg.build_ref,
        "port": cfg.api_port,
        "uptime_s": round(time.time() - _STARTED_AT, 2),
        "init_ms": RUNTIME.init_ms,
        "runtime_db": RUNTIME.runtime_db,
        "perf_enabled": RUNTIME.perf.enabled,
        "checks": checks,
        "warmup": RUNTIME.warmup,
    }
    return body, (200 if ready else 503)


def _dir_writable(path: str, label: str) -> dict[str, Any]:
    """写探针：真写一个临时文件再删（比 stat 更可信，能暴露只读挂载）。"""
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".deploy_health_probe"
        probe.write_text(str(time.time()), encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"ok": True, "path": str(p), "label": label, "detail": "可读写"}
    except OSError as exc:
        return {"ok": False, "path": str(path), "label": label, "reason": f"{type(exc).__name__}: {exc}"}


def _writable(path: Path) -> bool:
    """轻量写探针（供运行期 sqlite 选目录用），失败只返回 False 不抛。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("1", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _dir_readable(path: str, label: str) -> dict[str, Any]:
    """读探针：列目录条目数，确认索引卷真的挂进来了。"""
    try:
        p = Path(path)
        if not p.exists():
            return {"ok": False, "path": str(p), "label": label, "reason": "目录不存在"}
        entries = sorted(x.name for x in p.iterdir())
        return {"ok": True, "path": str(p), "label": label, "entries": len(entries),
                "sample": entries[:8], "detail": "可读"}
    except OSError as exc:
        return {"ok": False, "path": str(path), "label": label, "reason": f"{type(exc).__name__}: {exc}"}


@app.get("/", summary="服务信息")
def root() -> dict[str, Any]:
    cfg = RUNTIME.deploy
    return {
        "service": "金融问答系统（RAG）容器 API（工单17 优化后）",
        "work_order": WORK_ORDER,
        "api_version": API_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "containerized": _in_container(),
        "perf_enabled": RUNTIME.perf.enabled,
        "endpoints": ["GET /health", "GET /healthz", "GET /readyz", "GET /metrics",
                      "POST /api/v1/ask", "GET /api/v1/sessions", "GET /api/v1/index",
                      "POST /api/v1/retrieve", "POST /api/v1/uploads", "GET /api/v1/tasks",
                      "GET /api/v1/tasks/{id}", "POST /api/v1/admin/unload",
                      "GET /api/v1/perf", "GET /docs", "GET /openapi.json"],
        "ports": {"api": cfg.api_port, "ui": cfg.ui_port},
        "session_backend": (RUNTIME.sessions.backend if RUNTIME.sessions else "未初始化"),
        "sessions_on_disk": (RUNTIME.sessions.file.count() if RUNTIME.sessions else 0),
        "uptime_s": round(time.time() - _STARTED_AT, 2),
    }


def _in_container() -> bool:
    """判容器：``/.dockerenv`` 存在，或 cgroup 里出现 docker/kubepods/containerd。"""
    try:
        if Path("/.dockerenv").exists():
            return True
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            text = cgroup.read_text(encoding="utf-8", errors="ignore")
            return any(k in text for k in ("docker", "kubepods", "containerd"))
    except OSError:
        pass
    return False


# ====================================================================== #
# 核心问答接口（工单17：异步路由 + 入口限流）
# ====================================================================== #
def _ask_sync(payload: AskRequest) -> AskResponse:
    """问答主体（**同步**，由 ``run_in_threadpool`` 或直调执行；业务逻辑与工单10 一致）。"""
    if RUNTIME.engine is None:
        raise HTTPException(status_code=503, detail=f"服务未就绪：{RUNTIME.error or '引擎未初始化'}")
    session_id = (payload.session_id or "").strip() or None
    started = time.perf_counter()
    try:
        result = RUNTIME.engine.answer(
            payload.question, lang=payload.lang, top_k=payload.top_k,
            session_id=session_id, return_evidence=payload.with_evidence,
        )
    except IndexMissingError as exc:
        raise HTTPException(status_code=503, detail=f"索引缺失：{exc}") from exc
    except LLMProbeError as exc:
        raise HTTPException(status_code=502, detail=f"生成后端不可用：{exc}") from exc
    except RagError as exc:
        raise HTTPException(status_code=500, detail=f"处理失败：{exc}") from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    turns = 0
    backend = ""
    if RUNTIME.sessions is not None:
        sid = session_id or result.session_id or "anonymous"
        data = RUNTIME.sessions.append(sid, Turn(
            role="assistant", question=payload.question, answer=result.answer,
            refused=bool(result.refused), cited_pages=list(result.cited_pages or []),
            trace_id=result.trace_id, ts=time.time(),
        ))
        turns = len(data.get("turns") or [])
        backend = RUNTIME.sessions.backend
        session_id = sid

    with _METRICS_LOCK:
        _METRICS["rag_api_ask_total"] += 1
        _METRICS["rag_api_ask_latency_ms_sum"] += elapsed_ms
        _METRICS["rag_api_ask_latency_ms_sum_sq"] += elapsed_ms * elapsed_ms
        _METRICS["rag_api_ask_latency_ms_max"] = max(_METRICS["rag_api_ask_latency_ms_max"], elapsed_ms)
        if result.refused:
            _METRICS["rag_api_refused_total"] += 1

    RUNTIME._info("api.ask", trace_id=result.trace_id, session_id=session_id,
                  refused=bool(result.refused), elapsed_ms=elapsed_ms,
                  retrieve_ms=result.latency_ms.get("retrieve"),
                  generate_ms=result.latency_ms.get("generate"),
                  citations=len(result.citations or []), session_turns=turns)

    return AskResponse(
        question=result.question, answer=result.answer, refused=bool(result.refused),
        refusal_reason=result.refusal_reason, citations=list(result.citations or []),
        cited_pages=list(result.cited_pages or []), latency_ms=dict(result.latency_ms or {}),
        backend=result.backend, model=result.model, generation_method=result.generation_method,
        language=result.language, trace_id=result.trace_id, session_id=session_id,
        cache=dict(result.cache or {}), chunks=list(result.chunks or []),
        evidence=list(result.evidence or []), session_turns=turns, session_backend=backend,
    )


def _busy_response(exc: Exception, gate: Any) -> JSONResponse:
    """统一的 429 响应体（含在途/排队/上限，便于客户端与运维判断）。"""
    snap = getattr(exc, "snapshot", {}) or gate.snapshot()
    retry = gate.retry_after_s()
    with _METRICS_LOCK:
        if isinstance(exc, GateTimeout):
            _METRICS["rag_api_gate_timeout_total"] += 1
        else:
            _METRICS["rag_api_rejected_total"] += 1
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc), "inflight": snap.get("inflight"),
                 "queued": snap.get("queued"), "max_inflight": snap.get("max_inflight"),
                 "queue_max": snap.get("queue_max"), "retry_after_s": retry},
        headers={"Retry-After": str(retry)},
    )


@app.post("/api/v1/ask", response_model=AskResponse, summary="金融问答（核心接口）")
async def ask(payload: AskRequest) -> Any:
    """单次问答。会话 id 相同即构成多轮（历史落会话卷 + Redis，容器间共享）。

    工单17：
        * **F3** 先过有界闸门 —— 在途超过 ``PERF_MAX_INFLIGHT`` 且等待位满 ⇒ 立即 ``429``，
          **不再让请求在服务内部无上限堆积**（基座的 ``max_concurrency`` 配置项零引用，等于没有限流）；
        * **F4** 业务跑在 ``run_in_threadpool`` 里，事件循环不被阻塞，
          高压下 ``/healthz``、``/readyz`` 仍能立刻响应；
        * ``PERF_ASYNC_ROUTES=0`` 时退回同步调用（仅用于对照实验）。
    """
    if RUNTIME.engine is None:
        raise HTTPException(status_code=503, detail=f"服务未就绪：{RUNTIME.error or '引擎未初始化'}")
    gate = RUNTIME.gate
    try:
        # ★★ 关键修正（**实测踩坑，由 测试/测试数据/限流验证.json 抓出**）★★
        #   排队等待是**阻塞**的，绝不能发生在事件循环上，也不能占用业务线程池。
        #   见文件头 ``_GATE_WAIT_POOL`` 的两个连环 bug 说明。
        if gate.enabled and RUNTIME.perf.async_routes:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(RUNTIME.gate_wait_pool, gate.acquire)
        else:
            gate.acquire()
    except (GateRejected, GateTimeout) as exc:
        return _busy_response(exc, gate)
    try:
        # 基座的 ``def ask(...)`` 由 FastAPI 丢进 anyio 线程池执行；
        # 这里用 ``run_in_threadpool`` 复现**同一条执行路径**（保证 PERF_ENABLED=0 时行为一致），
        # 区别只在于：入口限流与中间件现在跑在事件循环上，不被阻塞。
        # ``PERF_ASYNC_ROUTES=0`` 时改为事件循环内直调（仅用于「阻塞事件循环」的对照实验）。
        if RUNTIME.perf.enabled and not RUNTIME.perf.async_routes:
            return _ask_sync(payload)
        return await run_in_threadpool(_ask_sync, payload)
    finally:
        gate.release()


@app.post("/api/v1/retrieve", summary="只检索不生成（诊断）")
def retrieve(question: str = Query(..., min_length=1), top_k: int = Query(10, ge=1, le=50),
             modality: str | None = Query(None)) -> dict[str, Any]:
    if RUNTIME.engine is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    try:
        return RUNTIME.engine.retrieve_only(question, top_k=top_k, modality=modality)
    except RagError as exc:
        raise HTTPException(status_code=500, detail=f"处理失败：{exc}") from exc


@app.get("/api/v1/index", summary="索引元信息")
def index_info() -> dict[str, Any]:
    if RUNTIME.engine is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    try:
        bundle = RUNTIME.engine.bundle
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"索引不可用：{type(exc).__name__}: {exc}") from exc
    manifest = getattr(bundle, "manifest", {}) or {}
    return {
        "index_dir": str(getattr(bundle, "index_dir", "")),
        "chunks": int(getattr(bundle, "size", 0)),
        "vector_dim": int(getattr(bundle.vector_index, "dim", 0)),
        "faiss": getattr(bundle.vector_index, "faiss_index", None) is not None,
        "bm25_vocab": int(bundle.bm25.vocab_size()) if hasattr(bundle, "bm25") else 0,
        "chunk_by_modality": manifest.get("chunk_by_modality") or {},
        "chunk_by_corpus": manifest.get("chunk_by_corpus") or {},
        "embed_model": manifest.get("embed_model"),
        "built_at": manifest.get("built_at"),
        "run_id": manifest.get("run_id"),
    }


@app.get("/api/v1/sessions", summary="会话列表（容器间共享的可观测证据）")
def list_sessions(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    if RUNTIME.sessions is None:
        raise HTTPException(status_code=503, detail="会话存储未初始化")
    return {
        "backend": RUNTIME.sessions.backend,
        "degraded_reason": RUNTIME.sessions.degraded_reason,
        "session_dir": str(RUNTIME.deploy.session_dir),
        "count": RUNTIME.sessions.file.count(),
        "sessions": RUNTIME.sessions.recent(limit),
    }


@app.get("/api/v1/sessions/{session_id}", summary="指定会话的多轮历史")
def get_session(session_id: str, limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    if RUNTIME.sessions is None:
        raise HTTPException(status_code=503, detail="会话存储未初始化")
    turns = RUNTIME.sessions.history(session_id, limit=limit)
    if not turns:
        raise HTTPException(status_code=404, detail=f"会话不存在或已过期：{session_id}")
    return {"session_id": session_id, "backend": RUNTIME.sessions.backend,
            "turns": len(turns), "history": turns}


# ====================================================================== #
# 工单17 新增：上传解析（F9）
# ====================================================================== #
def _parse_multipart(content_type: str, body: bytes) -> tuple[str, bytes]:
    """极简 multipart 解析（只处理**一个** file part，够用即止）。

    为什么不用 FastAPI 的 ``UploadFile``：它依赖 ``python-multipart``，
    而本机 pip 通道不通、不允许新增依赖（共享约定 §2）。
    """
    marker = "boundary="
    if marker not in content_type:
        raise UploadRejected("multipart 缺少 boundary")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    if not boundary:
        raise UploadRejected("multipart boundary 为空")
    sep = b"--" + boundary.encode("utf-8")
    for part in body.split(sep):
        if b"Content-Disposition" not in part:
            continue
        head, found, payload = part.partition(b"\r\n\r\n")
        if not found or b'filename="' not in head:
            continue
        head_text = head.decode("utf-8", errors="replace")
        start = head_text.index('filename="') + len('filename="')
        filename = head_text[start:].split('"', 1)[0]
        data = payload[:-2] if payload.endswith(b"\r\n") else payload
        return filename, data
    raise UploadRejected("multipart 中找不到 file 字段")


def _extract_upload(request: Request, body: bytes) -> tuple[str, bytes]:
    """从请求里取出 ``(filename, content)``，兼容 multipart 与裸字节两种投递方式。"""
    ctype = request.headers.get("content-type") or ""
    if ctype.lower().startswith("multipart/form-data"):
        return _parse_multipart(ctype, body)
    filename = (request.query_params.get("filename")
                or request.headers.get("x-filename") or "upload.pdf")
    return filename, body


@app.post("/api/v1/uploads", summary="上传 PDF 并解析（工单17 新增）")
async def uploads(request: Request) -> Any:
    """上传 PDF。

    * 优化后（``PERF_ENABLED=1``）：校验 → 落盘 → 入**有界**解析队列 → 立即 ``202 + task_id``；
      队列满 ⇒ ``429``（不堆积）。解析由工作线程池消费 —— 对应 RAGFlow 的 Task Executor。
    * 优化前（``PERF_ENABLED=0``）：校验 → **在请求线程里同步解析完**才返回 ``200``；
      这正是原文描述的「API Server 被耗时解析阻塞」反模式的可复现实现，
      用于产出可对比的场景B 基准数据。
    """
    body = await request.body()
    try:
        filename, content = _extract_upload(request, body)
        with _METRICS_LOCK:
            _METRICS["rag_api_uploads_total"] += 1
        if RUNTIME.perf.enabled and RUNTIME.parse_queue is not None:
            task = RUNTIME.parse_queue.submit(filename, content, max_mb=32.0)
            RUNTIME._info("api.upload_queued", task_id=task.task_id, filename=filename,
                          size_bytes=task.size_bytes, queue_depth=RUNTIME.parse_queue.queue_depth)
            return JSONResponse(status_code=202, content={
                "task_id": task.task_id, "status": task.status,
                "queue_depth": RUNTIME.parse_queue.queue_depth,
                "filename": task.filename, "size_bytes": task.size_bytes,
                "submitted_at": task.submitted_at, "mode": "async-queue",
            })
        row = await run_in_threadpool(
            parse_sync, filename, content, cfg=RUNTIME.perf,
            upload_dir=Path(RUNTIME.deploy.upload_dir), log_dir=Path(RUNTIME.deploy.log_dir))
        RUNTIME._info("api.upload_sync_done", task_id=row["task_id"], filename=filename,
                      status=row["status"], elapsed_ms=row["elapsed_ms"])
        return JSONResponse(status_code=(200 if row["status"] == "done" else 422),
                            content={**row, "mode": "sync-in-request"})
    except UploadRejected as exc:
        with _METRICS_LOCK:
            _METRICS["rag_api_upload_rejected_total"] += 1
        headers = {"Retry-After": "5"} if exc.status == 429 else None
        return JSONResponse(status_code=exc.status,
                            content={"detail": exc.message, **exc.extra}, headers=headers)
    except Exception as exc:  # noqa: BLE001 —— 任何意外都转 400，服务不崩
        with _METRICS_LOCK:
            _METRICS["rag_api_upload_rejected_total"] += 1
        return JSONResponse(status_code=400,
                            content={"detail": f"上传处理失败：{type(exc).__name__}: {exc}"})


@app.get("/api/v1/tasks", summary="解析任务列表与队列深度（工单17 新增）")
def tasks_list(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    if RUNTIME.parse_queue is None:
        return {"mode": "sync-in-request",
                "note": "PERF_ENABLED=0：无队列，解析在请求线程内同步完成",
                "queue_depth": 0, "queue_max": 0, "workers": 0,
                "counts": {"pending": 0, "running": 0, "done": 0, "failed": 0}, "tasks": []}
    stats = RUNTIME.parse_queue.stats()
    return {**stats, "tasks": RUNTIME.parse_queue.recent(limit)}


@app.get("/api/v1/tasks/{task_id}", summary="解析任务状态（工单17 新增）")
def task_detail(task_id: str) -> dict[str, Any]:
    if RUNTIME.parse_queue is None:
        raise HTTPException(status_code=404,
                            detail="PERF_ENABLED=0：无任务队列（解析同步完成，不产生 task_id）")
    task = RUNTIME.parse_queue.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    return task.to_dict()


# ====================================================================== #
# 工单17 新增：运维端点
# ====================================================================== #
@app.post("/api/v1/admin/unload", summary="主动卸载 Ollama 模型（工单17 新增）")
def admin_unload(request: Request) -> dict[str, Any]:
    """``keep_alive="0s"`` 主动卸载 —— 对应原文「GPU 内存释放不及时」。仅允许回环地址。"""
    client = request.client.host if request.client else ""
    if client not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail=f"仅允许本机调用（来源 {client}）")
    llm_cfg = getattr(get_config(), "llm", None)
    base_url = getattr(llm_cfg, "ollama_base_url", "http://127.0.0.1:11434")
    result = unload_models(base_url)
    RUNTIME.unload_count += 1
    RUNTIME.last_unload = {**result, "at": time.time()}
    RUNTIME._info("api.unload", **{k: v for k, v in result.items() if k != "errors"})
    return {**result, "count": RUNTIME.unload_count,
            "note": "已请求 Ollama 立即卸载；GET /api/ps 应随即为空"}


@app.get("/api/v1/perf", summary="性能自观测快照（工单17 新增）")
def perf_snapshot() -> dict[str, Any]:
    """把限流/缓存/连接池/单例/解析队列的运行时状态一次给全（运维与报告取证用）。"""
    engine_stats: dict[str, Any] = {}
    try:
        if RUNTIME.engine is not None:
            engine_stats = RUNTIME.engine.stats()
    except Exception as exc:  # noqa: BLE001
        engine_stats = {"error": f"{type(exc).__name__}: {exc}"}
    resident: list[str] = []
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3) as resp:
            resident = [str(m.get("name")) for m in (json.loads(resp.read()).get("models") or [])]
    except Exception:  # noqa: BLE001
        resident = []
    return {
        "perf_enabled": RUNTIME.perf.enabled,
        "perf_config": RUNTIME.perf.to_dict(),
        "limiter": RUNTIME.gate.snapshot(),
        "answer_cache": {k: v for k, v in engine_stats.items()
                         if k.startswith(("memory_cache", "asked", "cache_hit", "refused"))},
        "http_pool": get_http_pool().stats(),
        "singletons": registry_stats(RUNTIME.perf),
        "task_queue": (RUNTIME.parse_queue.stats() if RUNTIME.parse_queue is not None
                       else {"mode": "sync-in-request"}),
        "storage": engine_stats.get("storage"),
        "ollama": {"keep_alive": RUNTIME.perf.ollama_keep_alive, "resident": resident,
                   "unload_count": RUNTIME.unload_count, "last_unload": RUNTIME.last_unload},
    }


# ====================================================================== #
# 指标
# ====================================================================== #
@app.get("/metrics", summary="Prometheus 指标（文本格式）")
def metrics() -> PlainTextResponse:
    """Prometheus 文本 exposition 格式，供 compose 里的 prometheus 服务抓取。"""
    if not RUNTIME.deploy.metrics_enabled:
        raise HTTPException(status_code=404, detail="指标已关闭（DEPLOY_METRICS_ENABLED=0）")
    with _METRICS_LOCK:
        snap = dict(_METRICS)
    asks = max(snap["rag_api_ask_total"], 1.0)
    gate = RUNTIME.gate.snapshot()
    lines = [
        "# HELP rag_api_requests_total HTTP 请求总数",
        "# TYPE rag_api_requests_total counter",
        f"rag_api_requests_total {int(snap['rag_api_requests_total'])}",
        "# HELP rag_api_errors_total HTTP 4xx/5xx 总数",
        "# TYPE rag_api_errors_total counter",
        f"rag_api_errors_total {int(snap['rag_api_errors_total'])}",
        "# HELP rag_api_ask_total 问答请求总数",
        "# TYPE rag_api_ask_total counter",
        f"rag_api_ask_total {int(snap['rag_api_ask_total'])}",
        "# HELP rag_api_refused_total 拒答（不清楚）总数",
        "# TYPE rag_api_refused_total counter",
        f"rag_api_refused_total {int(snap['rag_api_refused_total'])}",
        "# HELP rag_api_ask_latency_ms 问答耗时（毫秒）",
        "# TYPE rag_api_ask_latency_ms summary",
        f"rag_api_ask_latency_ms_sum {snap['rag_api_ask_latency_ms_sum']:.2f}",
        f"rag_api_ask_latency_ms_count {int(snap['rag_api_ask_total'])}",
        f"rag_api_ask_latency_ms_max {snap['rag_api_ask_latency_ms_max']:.2f}",
        f"rag_api_ask_latency_ms_avg {snap['rag_api_ask_latency_ms_sum'] / asks:.2f}",
        # —— 工单17 新增：限流 ——
        "# HELP rag_api_inflight 当前在途请求数",
        "# TYPE rag_api_inflight gauge",
        f"rag_api_inflight {gate['inflight']}",
        "# HELP rag_api_queued 当前排队请求数",
        "# TYPE rag_api_queued gauge",
        f"rag_api_queued {gate['queued']}",
        "# HELP rag_api_inflight_peak 在途峰值",
        "# TYPE rag_api_inflight_peak gauge",
        f"rag_api_inflight_peak {gate['inflight_peak']}",
        "# HELP rag_api_rejected_total 因过载被拒（429）总数",
        "# TYPE rag_api_rejected_total counter",
        f"rag_api_rejected_total {int(snap['rag_api_rejected_total'])}",
        "# HELP rag_api_gate_timeout_total 排队超时总数",
        "# TYPE rag_api_gate_timeout_total counter",
        f"rag_api_gate_timeout_total {int(snap['rag_api_gate_timeout_total'])}",
        "# HELP rag_api_max_inflight 限流在途上限（配置）",
        "# TYPE rag_api_max_inflight gauge",
        f"rag_api_max_inflight {gate['max_inflight']}",
        # —— 工单17 新增：上传/解析 ——
        "# HELP rag_api_uploads_total 上传请求总数",
        "# TYPE rag_api_uploads_total counter",
        f"rag_api_uploads_total {int(snap['rag_api_uploads_total'])}",
        "# HELP rag_api_upload_rejected_total 上传被拒总数",
        "# TYPE rag_api_upload_rejected_total counter",
        f"rag_api_upload_rejected_total {int(snap['rag_api_upload_rejected_total'])}",
        "# HELP rag_api_uptime_seconds 进程存活秒数",
        "# TYPE rag_api_uptime_seconds gauge",
        f"rag_api_uptime_seconds {time.time() - _STARTED_AT:.2f}",
    ]
    pool_stats = get_http_pool().stats()
    lines += [
        "# HELP rag_http_pool_created_total 新建连接数",
        "# TYPE rag_http_pool_created_total counter",
        f"rag_http_pool_created_total {pool_stats['created_total']}",
        "# HELP rag_http_pool_reused_total 复用连接数",
        "# TYPE rag_http_pool_reused_total counter",
        f"rag_http_pool_reused_total {pool_stats['reused_total']}",
        "# HELP rag_http_pool_idle 空闲连接数",
        "# TYPE rag_http_pool_idle gauge",
        f"rag_http_pool_idle {sum(h.get('idle', 0) for h in pool_stats['hosts'].values())}",
    ]
    if RUNTIME.parse_queue is not None:
        q = RUNTIME.parse_queue.stats()
        lines += [
            "# HELP rag_parse_queue_depth 解析队列深度",
            "# TYPE rag_parse_queue_depth gauge",
            f"rag_parse_queue_depth {q['queue_depth']}",
            "# HELP rag_parse_tasks_total 解析任务计数",
            "# TYPE rag_parse_tasks_total counter",
            f'rag_parse_tasks_total{{status="done"}} {q["done"]}',
            f'rag_parse_tasks_total{{status="failed"}} {q["failed"]}',
            f'rag_parse_tasks_total{{status="rejected"}} {q["rejected"]}',
        ]
    if RUNTIME.engine is not None:
        try:
            stats = RUNTIME.engine.stats()
            lines += [
                "# HELP rag_engine_asked_total 引擎累计问答数",
                "# TYPE rag_engine_asked_total counter",
                f"rag_engine_asked_total {int(stats.get('asked', 0))}",
                "# HELP rag_engine_cache_hit_total 引擎缓存命中数",
                "# TYPE rag_engine_cache_hit_total counter",
                f"rag_engine_cache_hit_total {int(stats.get('cache_hit', 0))}",
                "# HELP rag_answer_cache_entries 答案缓存条目数",
                "# TYPE rag_answer_cache_entries gauge",
                f"rag_answer_cache_entries {int(stats.get('memory_cache', 0))}",
                "# HELP rag_answer_cache_evictions_total 答案缓存淘汰总数",
                "# TYPE rag_answer_cache_evictions_total counter",
                f"rag_answer_cache_evictions_total {int(stats.get('memory_cache_evictions', 0))}",
                "# HELP rag_answer_cache_expired_total 答案缓存过期总数",
                "# TYPE rag_answer_cache_expired_total counter",
                f"rag_answer_cache_expired_total {int(stats.get('memory_cache_expired', 0))}",
            ]
        except Exception:  # noqa: BLE001
            pass
    for name, item in (registry_stats(RUNTIME.perf).get("probe_caches") or {}).items():
        safe = name.replace("-", "_")
        lines += [
            f"# HELP rag_singleton_probe_hits_total 后端探测缓存命中（{name}）",
            "# TYPE rag_singleton_probe_hits_total counter",
            f'rag_singleton_probe_hits_total{{cache="{safe}"}} {item.get("hits", 0)}',
            f'rag_singleton_probe_misses_total{{cache="{safe}"}} {item.get("misses", 0)}',
        ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


# ====================================================================== #
# 异常兜底：任何未捕获异常都转成 JSON，绝不裸崩（共享约定 §4.4）
# ====================================================================== #
@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    with _METRICS_LOCK:
        _METRICS["rag_api_errors_total"] += 1
    try:
        if RUNTIME.log is not None:
            RUNTIME.log.log_event("http.unhandled", level="ERROR", path=request.url.path,
                                  error=f"{type(exc).__name__}: {exc}")
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse(status_code=500,
                        content={"detail": f"服务内部错误：{type(exc).__name__}", "path": request.url.path})


def main() -> None:   # pragma: no cover —— 便捷入口：python -m service.api
    import uvicorn

    cfg = get_deploy_config()
    uvicorn.run("service.api:app", host=cfg.api_host, port=cfg.api_port,
                log_level=cfg.log_level.lower(), workers=1)


if __name__ == "__main__":   # pragma: no cover
    main()
