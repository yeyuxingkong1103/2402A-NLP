# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

容器对外 FastAPI 服务：``研发/service/api.py``。

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
``/metrics``                  GET     Prometheus 文本格式指标（供 prometheus 服务抓取）
``/api/v1/ask``               POST    金融问答（核心接口）
``/api/v1/sessions``          GET     卷/Redis 里的会话列表（容器间共享的可观测证据）
``/api/v1/sessions/{id}``     GET     指定会话的多轮历史
``/api/v1/index``             GET     索引元信息（块数/维度/模态分布）
``/api/v1/retrieve``          POST    只检索不生成（诊断）
``/openapi.json``             GET     自动生成 OpenAPI 3.1（FastAPI 自带）
============================  ======  ==================================================

设计纪律：
    * 业务逻辑**零复制**：全部转发到 ``app.core.qa_engine.QAEngine``（工单7 基座）；
    * 会话落盘 + Redis 双写（验收项「容器间数据共享」），见 :mod:`service.session_store`；
    * 每个请求一行 JSON 日志（``DEPLOY_LOG_JSON=1``），写进日志卷 —— 容器运行期「无异常日志」
      这一条验收就是靠这份日志逐行核对的；
    * 健康检查**不伪装**：未就绪项如实给 ``ok=false`` 与原因字符串；
    * 所有阈值来自环境变量（:mod:`service.config`），代码里不写死端口/超时/并发数。
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# —— 让容器内 ``uvicorn service.api:app`` 能 import 到 app 包（/app/service → /app/app）——
if __package__ in (None, ""):   # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import describe_runtime, get_config          # noqa: E402
from app.core.errors import IndexMissingError, LLMProbeError, RagError   # noqa: E402
from app.core.logging_conf import get_logger, setup_logging        # noqa: E402
from app.core.qa_engine import QAEngine                            # noqa: E402
from app.storage.sqlite_manager import SqliteManager               # noqa: E402

from .config import DeployConfig, get_deploy_config                # noqa: E402
from .session_store import SharedSessionStore, Turn                # noqa: E402

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"
API_VERSION = "1.0.0"

_STARTED_AT = time.time()
_METRICS: dict[str, float] = {
    "rag_api_requests_total": 0.0,
    "rag_api_errors_total": 0.0,
    "rag_api_ask_total": 0.0,
    "rag_api_refused_total": 0.0,
    "rag_api_ask_latency_ms_sum": 0.0,
    "rag_api_ask_latency_ms_max": 0.0,
}
_METRICS_LOCK = threading.Lock()


# ====================================================================== #
# 请求/响应模型
# ====================================================================== #
class AskRequest(BaseModel):
    """``POST /api/v1/ask`` 请求体。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题（中文/英文）")
    session_id: str | None = Field(None, max_length=128, description="会话 id；多轮追问必传同一 id")
    lang: str = Field("auto", pattern="^(auto|zh|en)$", description="语言路径")
    top_k: int | None = Field(None, ge=1, le=20, description="检索条数（默认取 RAG_RETRIEVAL__TOP_K）")
    with_evidence: bool = Field(False, description="是否返回证据块（诊断用，响应体更大）")


class AskResponse(BaseModel):
    """``POST /api/v1/ask`` 响应体（字段与工单7 的 ``Answer`` 对齐）。"""

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
    """进程级运行时：QAEngine（业务）+ SharedSessionStore（共享会话）+ 日志句柄。"""

    def __init__(self, cfg: DeployConfig) -> None:
        self.deploy = cfg
        self.engine: QAEngine | None = None
        self.sessions: SharedSessionStore | None = None
        self.log: Any = None
        self.error: str = ""
        self.warmup: dict[str, Any] = {}
        self.init_ms: float = 0.0
        self.runtime_db: str = ""
        self._ready = threading.Event()

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

        # 运行期 sqlite（问答历史/答案缓存）必须落到**可写卷**：
        # 索引卷是 `:ro` 只读挂载，索引目录里写的 qa_runtime.sqlite3 在容器里必然失败，
        # 所以优先用 DEPLOY_RUNTIME_DIR，并做一次真实写探针确认（见 部署/过程问题记录.md 问题5）。
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
                   runtime_db=self.runtime_db, session_backend=self.sessions.backend)

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

    # -- 查询 ---------------------------------------------------------- #
    @property
    def ready(self) -> bool:
        return self.engine is not None and self._ready.is_set()


RUNTIME = Runtime(get_deploy_config())


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


app = FastAPI(
    title=RUNTIME.deploy.api_title,
    version=API_VERSION,
    description=f"工单编号：{WORK_ORDER} —— 金融问答系统容器化对外 API（RAG 基座：工单7）",
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
    body, code = _health_payload()
    return JSONResponse(status_code=(200 if code == 200 else 503), content=body)


@app.get("/health", summary="健康检查：索引 / LLM / Redis / 卷")
def health() -> JSONResponse:
    """健康检查。始终 200（诚实报告每一子项的 ok），便于运维一眼看清降级原因。"""
    body, _ = _health_payload()
    return JSONResponse(status_code=200, content=body)


def _health_payload() -> tuple[dict[str, Any], int]:
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
        "service": "金融问答系统（RAG）容器 API",
        "work_order": WORK_ORDER,
        "api_version": API_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "containerized": _in_container(),
        "endpoints": ["GET /health", "GET /healthz", "GET /readyz", "GET /metrics",
                      "POST /api/v1/ask", "GET /api/v1/sessions", "GET /api/v1/index",
                      "POST /api/v1/retrieve", "GET /docs", "GET /openapi.json"],
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
# 核心问答接口
# ====================================================================== #
@app.post("/api/v1/ask", response_model=AskResponse, summary="金融问答（核心接口）")
def ask(payload: AskRequest) -> AskResponse:
    """单次问答。会话 id 相同即构成多轮（历史落会话卷 + Redis，容器间共享）。"""
    if RUNTIME.engine is None:
        raise HTTPException(status_code=503, detail=f"服务未就绪：{RUNTIME.error or '引擎未初始化'}")
    cfg = RUNTIME.deploy
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
        "# HELP rag_api_uptime_seconds 进程存活秒数",
        "# TYPE rag_api_uptime_seconds gauge",
        f"rag_api_uptime_seconds {time.time() - _STARTED_AT:.2f}",
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
            ]
        except Exception:  # noqa: BLE001
            pass
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
