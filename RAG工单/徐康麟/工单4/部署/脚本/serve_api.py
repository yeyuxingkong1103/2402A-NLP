# -*- coding: utf-8 -*-
"""部署/脚本/serve_api.py —— FastAPI 服务（工单4 部署环节）。

契约来源：``设计/接口设计.md`` §3（端点总表与请求/响应体）。本文件**只做 HTTP 适配**：
检索/生成/引用/日志/缓存全部复用 ``研发/app/core/qa_engine.py::QAEngine``，
不复制任何业务逻辑；中文问答页直接复用 ``研发/app/ui/serve_fallback.py`` 的页面模板。

端点（FastAPI，OpenAPI 文档在 ``/docs``）：

    GET  /health        健康检查（依赖逐项给 ok/reason；未就绪 → 503，不伪装）
    POST /ask           单次问答（== /api/ask /api/query，兼容兜底 Web 页面）
    POST /ask/stream    流式问答（SSE；见下方「诚实声明」）
    POST /retrieve      只检索不生成（诊断/检索精度留痕用）
    GET  /questions     示例问题清单（含第 5、6 题）
    GET  /eval          评测端点说明（不跑评测）
    POST /eval          触发 16 题 golden 评测（同步；已有评测在跑 → 409）
    GET  /              中文 Web 问答页

**诚实声明（不得省略）**：
1. ``/ask/stream`` 的「流式」是**切片下发**（引擎一次性返回完整答案后按字符切片推 SSE），
   因此首个 ``delta`` 的到达时间**不是**首字延迟；真正的首字延迟取
   ``latency_ms.first_token``（由引擎实测），响应体里也带 ``first_token_ms``。
   事件里显式给出 ``stream_mode=post_hoc_slice`` 以免被误读为真流式。
2. ``/eval`` 的判分与命中判定**复用** ``测试/common/golden.py``（确定性，非 LLM 自评）；
   响应里给出 golden 文件路径与 sha256，便于核对是否同一版真值。

启动（推荐统一入口，保证解释器与 UTF-8 环境一致）：

    pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode api -Port 8600

或直接：

    pwsh -NoProfile -File run_py.ps1 部署/脚本/serve_api.py --host 127.0.0.1 --port 8600

退出码：``0`` 正常 / ``2`` 参数错误 / ``3`` 索引缺失 / ``4`` 依赖缺失（fastapi、uvicorn 不可用）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- 仓库根 & import 路径
def _find_repo_root(start: Path) -> Path:
    """向上找仓库根（同时具备 run_py.ps1 与 研发/app 的目录）。"""
    for cand in (start, *start.parents):
        if (cand / "run_py.ps1").is_file() and (cand / "研发" / "app").is_dir():
            return cand
    raise RuntimeError(f"找不到仓库根（从 {start} 向上未发现 run_py.ps1 + 研发/app）")


REPO_ROOT: Path = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT / "研发") not in sys.path:                     # 与 研发/app/main.py 同一约定
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from fastapi import FastAPI, Request                                          # noqa: E402
from fastapi.exceptions import RequestValidationError                         # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse   # noqa: E402
from pydantic import BaseModel, Field                                         # noqa: E402

from app.core.config import describe_runtime, get_config                      # noqa: E402
from app.core.errors import IndexMissingError, LLMProbeError, RagError        # noqa: E402
from app.core.logging_conf import setup_logging                               # noqa: E402
from app.core.qa_engine import QAEngine                                       # noqa: E402
from app.storage.sqlite_manager import SqliteManager                          # noqa: E402

VERSION = "1.0.0"
SERVICE = "部署/脚本/serve_api.py（FastAPI）"
EXIT_OK, EXIT_PARAM, EXIT_INDEX, EXIT_DEPS = 0, 2, 3, 4

ENDPOINTS = ["GET /", "GET /health", "POST /ask", "POST /ask/stream", "POST /retrieve",
             "GET /questions", "GET /eval", "POST /eval", "GET /docs"]

# --------------------------------------------------------------------------- 引擎（进程内单例）
_STATE: dict[str, Any] = {"engine": None, "warmup": None, "error": None}
_STATE_LOCK = threading.Lock()
_EVAL_LOCK = threading.Lock()


def build_engine(*, warmup: bool = True) -> tuple[Any, Any]:
    """构造（并缓存）QAEngine；失败时把异常记在 ``_STATE['error']``，由 /health 如实报告。"""
    with _STATE_LOCK:
        if _STATE["engine"] is not None or _STATE["error"] is not None:
            return _STATE["engine"], _STATE["warmup"]
        cfg = get_config()
        log = setup_logging(cfg)
        log.log_event("serve_api.start", version=VERSION, runtime=describe_runtime(),
                      endpoints=ENDPOINTS, warmup=warmup)
        storage = None
        try:
            storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
        except Exception as exc:  # noqa: BLE001 —— 持久化失败不阻断问答（显式降级）
            log.log_event("serve_api.storage_failed", level="WARNING",
                          error=f"{type(exc).__name__}: {exc}", action="问答继续，不落盘历史")
        engine = QAEngine(cfg=cfg, logger=log, storage=storage)
        if warmup:
            try:
                _STATE["warmup"] = engine.warmup(with_generate=True)
            except IndexMissingError as exc:
                _STATE["error"] = exc
                log.log_event("serve_api.warmup_failed", level="ERROR",
                              error=f"{type(exc).__name__}: {exc}")
        _STATE["engine"] = engine
        return _STATE["engine"], _STATE["warmup"]


def get_engine() -> Any:
    engine, _ = build_engine()
    if _STATE["error"] is not None:
        raise _STATE["error"]
    return engine


def _golden_module() -> Any:
    """加载 测试/common/golden.py（判分与命中口径的**唯一**真值实现）。"""
    path = REPO_ROOT / "测试" / "common" / "golden.py"
    if not path.is_file():
        raise FileNotFoundError(f"找不到判分库 {path}")
    spec = importlib.util.spec_from_file_location("golden_lib_serve_api", path)
    if spec is None or spec.loader is None:                     # pragma: no cover - 环境异常
        raise RuntimeError(f"无法加载判分库 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------- 请求体
class AskRequest(BaseModel):
    """`POST /ask` 请求体（设计/接口设计.md §3.3）。"""

    question: str = Field(..., min_length=1, max_length=500, description="1~500 字，非空白")
    lang: str = Field("auto", pattern="^(zh|en|auto)$")
    top_k: int | None = Field(None, ge=1, le=20)
    session_id: str | None = None
    return_evidence: bool = True
    use_cache: bool = True


class RetrieveRequest(BaseModel):
    """`POST /retrieve` 请求体（设计/接口设计.md §3.4）。"""

    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    modality: str | None = Field(None, pattern="^(text|table|image)$")


class EvalRequest(BaseModel):
    """`POST /eval` 请求体（工单4 16 题 golden 评测）。"""

    ids: list[int] | None = Field(None, description="只评测这些 golden id；缺省＝全 16 题")
    limit: int | None = Field(None, ge=1, le=16, description="只评测前 N 题（按 golden order）")
    top_k: int = Field(5, ge=1, le=20, description="检索返回条数；命中判定 K 与之一致")
    use_cache: bool = True
    include_details: bool = True


# --------------------------------------------------------------------------- 响应辅助
def _error(code: str, message: str, *, status: int, trace_id: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if trace_id:
        body["error"]["trace_id"] = trace_id
    return JSONResponse(body, status_code=status)


def _ask_payload(result: Any) -> dict[str, Any]:
    """Answer → JSON（并把评测端用的 cited_pages / chunks 一并给出，兼容 T6 客户端）。"""
    body = result.to_dict()
    body["cited_pages"] = result.cited_pages
    body["chunks"] = result.chunks
    return body


def _engine_error(exc: Exception) -> JSONResponse:
    """把引擎异常映射到设计文档约定的错误码。"""
    if isinstance(exc, IndexMissingError):
        return _error("E_INDEX_MISSING", str(exc), status=503)
    if isinstance(exc, LLMProbeError):
        return _error("E_BACKEND_UNAVAILABLE", str(exc), status=503)
    if isinstance(exc, RagError):
        return _error("E_INTERNAL", f"{type(exc).__name__}: {exc}", status=500)
    return _error("E_INTERNAL", f"{type(exc).__name__}: {exc}", status=500)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


# --------------------------------------------------------------------------- 应用
def create_app(*, warmup: bool = True) -> FastAPI:
    """构造 FastAPI 应用（``/docs`` 自动生成 OpenAPI）。"""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        build_engine(warmup=warmup)          # 启动即预热：首个请求不背负冷启动
        yield

    app = FastAPI(title="招股说明书图像内容 RAG 问答系统（工单4）", version=VERSION,
                  description="FastAPI 适配层：检索/生成/判分均复用研发与测试的既有实现。",
                  lifespan=lifespan)

    # ---------------------------------------------------------------- 中文 Web 页面
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        """中文问答页（复用 研发/app/ui/serve_fallback.py 的模板，页面 JS 调 /api/ask）。"""
        from app.ui import serve_fallback as ui

        html = (ui._PAGE_HTML.replace("__TITLE__", ui.TITLE)
                .replace("__DEMOS__", json.dumps(ui.DEMO_QUESTIONS, ensure_ascii=False)))
        return HTMLResponse(html)

    # ---------------------------------------------------------------- 健康检查
    @app.get("/health")
    @app.get("/api/health", include_in_schema=False)
    def health() -> JSONResponse:
        """依赖逐项体检；``status=ok`` → 200，否则 503 并给出缺哪一项（不伪装）。"""
        payload: dict[str, Any] = {"status": "degraded", "version": VERSION, "service": SERVICE,
                                   "endpoints": ENDPOINTS, "repo_root": str(REPO_ROOT)}
        try:
            engine = get_engine()
            payload.update(engine.health())
        except Exception as exc:  # noqa: BLE001 —— 必须如实报告，不得 ok:true 掩盖
            payload["checks"] = {"engine": {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}}
            payload["error"] = {"code": "E_INDEX_MISSING" if isinstance(exc, IndexMissingError)
                                else "E_INTERNAL", "message": str(exc)}
        warmup = _STATE["warmup"] or {}
        payload["warmed"] = _STATE["warmup"] is not None
        payload["warmup"] = {"index_chunks": warmup.get("index_chunks"),
                             "elapsed_ms": warmup.get("elapsed_ms"),
                             "generate_ok": bool((warmup.get("generate") or {}).get("ok"))}
        status = 200 if payload.get("status") == "ok" else 503
        return JSONResponse(payload, status_code=status)

    # ---------------------------------------------------------------- 问答
    @app.post("/ask")
    @app.post("/api/ask", include_in_schema=False)
    @app.post("/api/query", include_in_schema=False)
    def ask(req: AskRequest) -> JSONResponse:
        """单次问答；拒答也是 200（那是正确行为）。"""
        try:
            engine = get_engine()
            result = engine.answer(req.question, lang=req.lang, top_k=req.top_k,
                                   session_id=req.session_id,
                                   return_evidence=req.return_evidence, use_cache=req.use_cache)
        except Exception as exc:  # noqa: BLE001
            return _engine_error(exc)
        return JSONResponse(_ask_payload(result))

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> Any:
        """SSE 流式问答（**切片下发**，见模块文档「诚实声明」）。"""
        try:
            engine = get_engine()
            result = engine.answer(req.question, lang=req.lang, top_k=req.top_k,
                                   session_id=req.session_id,
                                   return_evidence=req.return_evidence, use_cache=req.use_cache)
        except Exception as exc:  # noqa: BLE001
            return _engine_error(exc)

        def gen():
            head = {"type": "citations", "citations": result.citations,
                    "cited_pages": result.cited_pages, "refused": result.refused,
                    "stream_mode": "post_hoc_slice",
                    "first_token_ms": result.latency_ms.get("first_token"),
                    "note": "切片下发：首个 delta 不是首字延迟，首字延迟取 first_token_ms"}
            yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"
            text = result.answer or ""
            step = 24
            for i in range(0, len(text), step):
                yield f"data: {json.dumps({'type': 'delta', 'text': text[i:i + step]}, ensure_ascii=False)}\n\n"
            done = {"type": "done", "refused": result.refused, "latency_ms": result.latency_ms,
                    "first_token_ms": result.latency_ms.get("first_token"),
                    "backend": result.backend, "model": result.model,
                    "generation_method": result.generation_method, "trace_id": result.trace_id}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store",
                                          "X-Stream-Mode": "post_hoc_slice"})

    # ---------------------------------------------------------------- 检索
    @app.post("/retrieve")
    @app.post("/api/retrieve", include_in_schema=False)
    def retrieve(req: RetrieveRequest) -> JSONResponse:
        """只检索不生成（检索精度留痕/调参用）。"""
        try:
            engine = get_engine()
            payload = engine.retrieve_only(req.question, top_k=req.top_k, modality=req.modality)
        except Exception as exc:  # noqa: BLE001
            return _engine_error(exc)
        return JSONResponse(payload)

    # ---------------------------------------------------------------- 问题清单
    @app.get("/questions")
    @app.get("/api/questions", include_in_schema=False)
    def questions() -> JSONResponse:
        from app.ui import serve_fallback as ui

        return JSONResponse({"questions": ui.DEMO_QUESTIONS, "count": len(ui.DEMO_QUESTIONS)})

    # ---------------------------------------------------------------- 评测
    @app.get("/eval")
    def eval_info() -> JSONResponse:
        """评测端点说明（不跑评测）——先让调用方看清口径再触发。"""
        golden_canonical = REPO_ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl"
        golden_mirror = REPO_ROOT / "测试" / "测试数据" / "golden_qa_16.jsonl"
        return JSONResponse({
            "endpoint": "POST /eval",
            "body": {"ids": "int[] 可选", "limit": "int 可选(1~16)", "top_k": 5,
                     "use_cache": True, "include_details": True},
            "judge": "测试/common/golden.py::judge_answer（确定性，非 LLM 自评）",
            "hit_rule": "测试/common/golden.py::check_retrieval_hit（前 top_k 条内命中证据原文）",
            "golden": {
                "canonical": str(golden_canonical),
                "canonical_sha256": _sha256(golden_canonical),
                "mirror": str(golden_mirror),
                "mirror_sha256": _sha256(golden_mirror),
            },
            "note": "同步执行：16 题全量约 15~30 s（缓存命中时更快）；已有评测在跑返回 409。",
        })

    @app.post("/eval")
    def eval_run(req: EvalRequest) -> JSONResponse:
        """跑 16 题 golden 评测（同步；判分复用 测试/common/golden.py）。"""
        if not _EVAL_LOCK.acquire(blocking=False):
            return _error("E_EVAL_BUSY", "已有评测在跑，请稍后重试（口径：同时只允许一个评测）", status=409)
        try:
            mod = _golden_module()
            records = list(mod.load_golden())
            if req.ids:
                wanted = {int(i) for i in req.ids}
                records = [r for r in records if int(r.get("id")) in wanted]
            if req.limit:
                records = records[: int(req.limit)]
            engine = get_engine()
            started = time.perf_counter()
            details: list[dict[str, Any]] = []
            walls: list[float] = []
            for record in records:
                t0 = time.perf_counter()
                result = engine.answer(str(record["question"]), lang="zh", top_k=req.top_k,
                                       return_evidence=True, use_cache=req.use_cache)
                wall = round((time.perf_counter() - t0) * 1000, 2)
                walls.append(wall)
                judge = mod.judge_answer(record, result.answer)
                hit = mod.check_retrieval_hit(record, result.chunks, top_k=req.top_k)
                item = {"id": record.get("id"), "order": record.get("order"),
                        "question": record["question"], "answer": result.answer,
                        "correct": bool(judge.get("correct")), "judge_reason": judge.get("reason"),
                        "hit_complete": bool(hit.get("hit")), "hit_top_k": hit.get("top_k"),
                        "refused": bool(result.refused), "refusal_reason": result.refusal_reason,
                        "cited_pages": result.cited_pages, "latency_wall_ms": wall,
                        "latency_server_ms": result.latency_ms,
                        "cache_hit": bool((result.cache or {}).get("hit")),
                        "generation_method": result.generation_method, "trace_id": result.trace_id}
                if req.include_details:
                    item["judge"] = judge
                    item["hit"] = {"matched": hit.get("matched"), "missed": hit.get("missed")}
                details.append(item)
            n = len(details)
            n_correct = sum(1 for d in details if d["correct"])
            n_hit = sum(1 for d in details if d["hit_complete"])
            golden_path = REPO_ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl"
            if not golden_path.is_file():
                golden_path = REPO_ROOT / "测试" / "测试数据" / "golden_qa_16.jsonl"
            summary = {
                "n_questions": n, "n_correct": n_correct,
                "accuracy": round(n_correct / n, 4) if n else None,
                "n_hit_complete": n_hit,
                "retrieval_hit_rate_complete": round(n_hit / n, 4) if n else None,
                "top_k": req.top_k, "use_cache": req.use_cache,
                "n_refused": sum(1 for d in details if d["refused"]),
                "n_cache_hit": sum(1 for d in details if d["cache_hit"]),
                "latency_wall_p50_ms": _percentile(walls, 0.50),
                "latency_wall_p95_ms": _percentile(walls, 0.95),
                "latency_wall_max_ms": max(walls) if walls else None,
                "n_latency_over_3000ms": sum(1 for w in walls if w > 3000.0),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            payload = {"summary": summary, "details": details,
                       "golden": {"path": str(golden_path), "sha256": _sha256(golden_path),
                                  "n_records_in_file": len(mod.load_golden())},
                       "judged_by": "测试/common/golden.py::judge_answer（确定性，非 LLM 自评）",
                       "hit_rule": "check_retrieval_hit（前 top_k 条 chunk 内命中证据原文）",
                       "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            return JSONResponse(payload)
        except Exception as exc:  # noqa: BLE001
            return _engine_error(exc)
        finally:
            _EVAL_LOCK.release()

    # ---------------------------------------------------------------- 错误映射（400/500）
    @app.exception_handler(RequestValidationError)
    async def _on_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = (exc.errors() or [{}])[0]
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        code = "E_PARAM_TOP_K_RANGE" if "top_k" in loc else "E_PARAM_QUESTION_EMPTY"
        return _error(code, f"参数校验失败：{loc} {first.get('msg', '')}", status=400)

    @app.exception_handler(Exception)
    async def _on_error(_request: Request, exc: Exception) -> JSONResponse:   # pragma: no cover
        return _error("E_INTERNAL", f"{type(exc).__name__}: {exc}", status=500)

    return app


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve_api.py", description="工单4 FastAPI 服务（设计/接口设计.md §3 契约）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8600, help="监听端口（默认 8600）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（仅调试；会引入冷启动延迟）")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning",
                                                                "info", "debug", "trace"])
    parser.add_argument("--print-config", action="store_true", help="打印运行时配置后退出（不启动服务）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_config:
        cfg = get_config()
        print(json.dumps({"repo_root": str(REPO_ROOT), "runtime": describe_runtime(),
                          "index_dir": str(cfg.paths.index_dir), "log_dir": str(cfg.paths.log_dir),
                          "host": args.host, "port": args.port}, ensure_ascii=False, indent=2))
        return EXIT_OK
    try:
        import uvicorn
    except ImportError as exc:                                   # pragma: no cover - 依赖体检
        print(f"[依赖缺失] 无法 import uvicorn：{exc}\n"
              f"本机禁止裸 pip install（会写坏只读的 工单1 venv）；"
              f"如需安装请用 --target 研发/第三方库，见 部署/README.md", file=sys.stderr)
        return EXIT_DEPS

    print(f"[OK] {SERVICE} 启动中：http://{args.host}:{int(args.port)}/  "
          f"（/health、/ask、/retrieve、/eval、/docs；Ctrl+C 停止）", flush=True)
    try:
        uvicorn.run(create_app(warmup=not args.no_warmup), host=args.host, port=int(args.port),
                    log_level=args.log_level)
    except KeyboardInterrupt:                                    # pragma: no cover
        print("\n[OK] 已停止 FastAPI 服务")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
