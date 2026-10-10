# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""FastAPI 服务：/health、/api/ask、/api/stats、/api/ingest。

压测直接打本服务（10/50/100 并发），与 Streamlit 前端解耦，
保证压测数据不被前端开销污染。

启动（工厂形式，避免导入即加载模型/锁索引）：
    uvicorn rag04.api.server:_get_app --factory --port 8000

时延口径：``latency_ms`` 一律取 ``Answer.latency_ms``（流水线内自测，不含
HTTP 与前端开销），并按 ``Settings.latency_budget_ms`` 如实给出
``within_budget``；超预算不截断、不缓存、不重试，缺口由 /health 与压测报告暴露。

运维告警（Task 19 复核记录，勿删）：
  - ``POST /api/ingest`` 是**同步、无鉴权、分钟级**的全量重建（``build(reset=True)``：
    先清空三个 collection 再重建，语义与端点 docstring 一致），且独占当前
    worker（本机实测全量约 15 分钟）。压测/演示窗口内误触发会直接污染数据：
    必须显式 ``{"confirm": true}``（否则 400）。重建是可达路径：会先释放本进程
    的存储句柄再让 ``build_all`` 独占索引（Qdrant 本地客户端对 data/qdrant 持
    独占锁，同进程/跨进程都一样）。同一时刻只允许一个重建（并发第二个 → 409），
    重建期间问答 → 503。标准重建路径仍是 CLI：
    ``python scripts/build_index.py full_04 --reset``。
  - 跨进程撞锁（别的进程握着 data/qdrant，例如 CLI 正在重建）时，/api/ask 与
    /api/ingest 都返回可执行的 503/409 文案并附原始错误，绝不抛裸栈。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag04.config import VALID_MODES, Settings, get_settings, latency_verdict

logger = logging.getLogger("rag04.api")

MAX_QUESTION_LEN = 1000

# 重建/撞锁的运维文案（错误契约：顶层 error 字段，见 _flat_http_error）
INGEST_CONFIRM_HINT = (
    "重建是同步阻塞的分钟级全量操作（本机约 15 分钟，会先清空三个 collection 再重建），"
    "期间该 worker 不响应其他请求。确认执行请发送 {\"confirm\": true}；"
    "常规重建建议改用 CLI：python scripts/build_index.py full_04 --reset"
)
INGEST_BUSY_HINT = (
    "重建进行中：同一时刻只允许一个重建任务（约 15 分钟），请等待其完成后重试"
)
INGEST_LOCK_HINT = (
    "索引已被占用，重建需先停止服务或改用 CLI"
    "（python scripts/build_index.py full_04 --reset）"
)
ASK_REBUILDING_HINT = (
    "索引重建进行中（同步全量，约 15 分钟），期间无法问答；"
    "可查 GET /health 的 rebuilding 字段，稍后重试"
)
ASK_LOCK_HINT = (
    "索引被其它进程占用（例如 CLI 正在重建），本进程暂时无法打开索引；"
    "请等待其完成后重试"
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LEN)
    mode: str | None = None      # 提示性字段：不切换链路，回显见 AskResponse.mode


class IngestRequest(BaseModel):
    """重建请求。``confirm`` 必须显式置 true（见模块 docstring 运维告警）。"""
    confirm: bool = False


class AskResponse(BaseModel):
    answer: str
    lang: str
    citations: list[dict[str, Any]]
    latency_ms: float
    llm_backend: str
    refused: bool = False
    # 3 秒硬指标的诚实上报：由 Answer.latency_ms 与 Settings.latency_budget_ms
    # 计算，不截断也不美化（实测 p50 ≈ 3.9s），供压测/界面直接断言与展示。
    latency_budget_ms: float
    within_budget: bool
    mode: str          # 实际生效模式：请求里的 mode 不切换链路，这里回显真实口径


def _settings_of(pipeline: Any) -> Settings:
    """应用生效配置：注入的流水线自带 Settings 就沿用，否则读全局配置。

    测试注入的是 MagicMock，其 ``.s`` 并非 Settings；此处退回全局配置，
    保证 /health 等端点拿到的是真实、可序列化、可断言的配置值。
    """
    s = getattr(pipeline, "s", None)
    return s if isinstance(s, Settings) else get_settings()


def _is_lock_error(exc: BaseException) -> bool:
    """是否为 Qdrant 本地客户端的独占锁错误（同/跨进程占用同一目录）。"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return ("already accessed" in text or "storage folder" in text
            or "already locked" in text)


def _warmup(pipeline: Any) -> dict[str, Any]:
    """启动预热：预载 reranker / CLIP / 嵌入模型，抹掉首次提问的加载时延。

    只在工厂路径（``create_app()`` 自行构造真实流水线，即 uvicorn --factory
    的启动路径）调用；注入流水线（测试）不调用。失败仅记录不抛出 —— 预热是
    优化，不是正确性前提。返回值同时进日志与 /health，说明预热是否真的跑过。
    """
    fn = getattr(pipeline, "warmup", None)
    if not callable(fn):
        logger.warning("流水线无 warmup()，跳过启动预热")
        return {"ran": False, "ok": False, "reason": "no_warmup_method"}

    t0 = time.perf_counter()
    try:
        fn()
    except Exception as e:          # 预热失败绝不能阻断服务启动
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        logger.warning("启动预热失败（不影响问答，仅首次提问变慢）：%s: %s",
                       type(e).__name__, e)
        return {"ran": True, "ok": False, "elapsed_ms": elapsed,
                "error": f"{type(e).__name__}: {e}"}

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("启动预热完成，耗时 %.1f ms（reranker/CLIP 常驻，首次问答不再付加载时延）",
                elapsed)
    return {"ran": True, "ok": True, "elapsed_ms": elapsed}


def create_app(pipeline=None) -> FastAPI:
    """构造应用。pipeline 可注入（测试用）；未注入时构造真实流水线并预热。"""
    app = FastAPI(title="工单04 图像内容解析及检索优化", version="1.0.0")

    if pipeline is None:
        from rag04.pipeline import RAGPipeline
        s = get_settings()
        # 日志由 RAGPipeline.__init__ → setup_logging(s) 统一配置（轮转文件 +
        # stderr）；"rag04.api" 的子记录会 propagate 到 "rag04" 落盘，无需重复配置。
        pipeline = RAGPipeline(s)
        app.state.warmup = _warmup(pipeline)
    else:
        s = _settings_of(pipeline)
        app.state.warmup = {"ran": False, "ok": False,
                            "reason": "pipeline_injected"}

    app.state.settings = s
    app.state.pipeline = pipeline
    # 重建是分钟级同步操作：进程内串行化（第二个并发重建 → 409），并让
    # /api/ask 在重建窗口内返回可执行的 503 而不是降级成「拒答」或裸 500。
    app.state.rebuild_lock = threading.Lock()
    app.state.rebuilding = False

    @app.exception_handler(HTTPException)
    async def _flat_http_error(_request, exc: HTTPException) -> JSONResponse:
        """把 dict 型 detail 摊平为响应体，错误契约统一为顶层 ``error`` 字段。

        FastAPI 默认包一层 {"detail": ...}；本项目（含压测与界面）按
        {"error": ...} 读取，故此处统一摊平；字符串 detail 保持默认形状。
        """
        content = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
        return JSONResponse(status_code=exc.status_code, content=content,
                            headers=getattr(exc, "headers", None))

    @app.get("/health")
    def health() -> dict:
        try:
            out = dict(app.state.pipeline.health())
        except Exception as e:
            logger.error("健康检查失败：%s", e)
            out = {"status": "degraded", "error": f"{type(e).__name__}: {e}"}
        out.setdefault("mode", app.state.settings.pipeline_mode)
        out["latency_budget_ms"] = app.state.settings.latency_budget_ms
        out["warmup"] = app.state.warmup
        out["rebuilding"] = bool(app.state.rebuilding)   # 重建窗口内问答会 503
        return out

    @app.post("/api/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        q = (req.question or "").strip()
        if not q:
            raise HTTPException(status_code=422, detail="问题不能为空")
        mode = app.state.settings.pipeline_mode
        # 空串/None 视为「未指定」（历史行为），只有真给了值才校验，避免误报 422
        if req.mode and req.mode not in VALID_MODES:
            raise HTTPException(
                status_code=422,
                detail={"error": f"mode 必须是 {list(VALID_MODES)} 之一，收到 {req.mode!r}"})
        if req.mode and req.mode != mode:
            # 一个服务进程只有一套索引与一套开关，请求里的 mode 不切换链路；
            # 实际口径由响应里的 mode 字段回显（日志同时留痕）。
            logger.warning("请求模式 %s 与运行模式 %s 不一致，按运行模式作答",
                           req.mode, mode)
        if app.state.rebuilding:
            # 重建会释放存储句柄：此刻作答要么撞锁要么静默降级成拒答，都不如实
            raise HTTPException(status_code=503, detail={"error": ASK_REBUILDING_HINT})
        try:
            ans = app.state.pipeline.ask(q)
        except Exception as e:
            logger.exception("问答失败")
            if _is_lock_error(e):
                raise HTTPException(
                    status_code=503,
                    detail={"error": f"{ASK_LOCK_HINT}（原始错误：{type(e).__name__}: {e}）"}
                ) from e
            if app.state.rebuilding:
                # 重建窗口内任何失败都可能源于索引被释放，统一按 503 上报
                raise HTTPException(
                    status_code=503,
                    detail={"error": f"{ASK_REBUILDING_HINT}（原始错误：{type(e).__name__}: {e}）"}
                ) from e
            raise HTTPException(
                status_code=500, detail={"error": f"{type(e).__name__}: {e}"}
            ) from e
        budget = app.state.settings.latency_budget_ms
        # 判定口径 = 上报口径（先取到 1 位小数再比较）：唯一实现在
        # rag04.config.latency_verdict，界面与压测报告共用同一函数，
        # 否则 3000.04ms 会显示成 3000.0 却标 within_budget=false，自相矛盾。
        latency, within_budget = latency_verdict(ans.latency_ms, budget)
        return AskResponse(
            answer=ans.answer, lang=ans.lang, citations=ans.citations,
            latency_ms=latency, llm_backend=ans.llm_backend,
            refused=ans.refused,
            latency_budget_ms=budget,
            within_budget=within_budget,
            mode=mode,
        )

    @app.get("/api/stats")
    def stats() -> dict:
        """库内计数。``store`` 字段区分「未加载」与「已加载但为空」两种状态。"""
        try:
            store = getattr(app.state.pipeline, "store", None)
            if store is None:
                # 绝不把「本次没打开存储」报成 {"counts": {}}——那与空索引无法区分
                return {"mode": app.state.settings.pipeline_mode,
                        "store": "not_loaded", "counts": None}
            return {"mode": app.state.settings.pipeline_mode,
                    "store": "ok", "counts": store.counts()}
        except Exception as e:
            return {"mode": app.state.settings.pipeline_mode,
                    "store": "error", "counts": None,
                    "error": f"{type(e).__name__}: {e}"}

    @app.post("/api/ingest")
    def ingest(req: IngestRequest | None = None) -> dict:
        """同步全量重建（``build(reset=True)``：先清空三个 collection 再重建）。

        危险操作：确认、串行化与撞锁防护见模块 docstring。

        这是可达路径：``RAGPipeline.build()`` 会先释放本进程的存储句柄，再让
        ``build_all`` 独占 data/qdrant（服务启动预热会一直持有该句柄，故不能
        简单地以「存储已打开」为由拒绝重建）。``reset=True`` 让端点语义与
        docstring 一致：不清空的话旧分块会与新分块共存，重建名不副实。
        """
        if not (req and req.confirm):
            logger.warning("拒绝未确认的 /api/ingest 调用（重建为分钟级阻塞操作）")
            raise HTTPException(status_code=400, detail={"error": INGEST_CONFIRM_HINT})
        if not app.state.rebuild_lock.acquire(blocking=False):
            logger.warning("拒绝重建：已有重建在进行中")
            raise HTTPException(status_code=409, detail={"error": INGEST_BUSY_HINT})

        try:
            # 闸门必须在 try 内：设置标志/写日志若在 try 外抛错，finally 不会执行，
            # rebuilding 与 rebuild_lock 永久卡住（此后所有问答 503、重建 409）。
            app.state.rebuilding = True
            logger.warning("开始同步全量重建：该 worker 将被独占约 15 分钟，"
                           "期间 /api/ask 返回 503")
            stats_list = app.state.pipeline.build(reset=True)
        except Exception as e:
            logger.exception("建库失败")
            if _is_lock_error(e):                    # 别的进程占着索引
                raise HTTPException(
                    status_code=409,
                    detail={"error": f"{INGEST_LOCK_HINT}（原始错误：{type(e).__name__}: {e}）"}
                ) from e
            raise HTTPException(
                status_code=500, detail={"error": f"{type(e).__name__}: {e}"}
            ) from e
        finally:
            app.state.rebuilding = False             # 失败也必须解除重建标记
            app.state.rebuild_lock.release()
        logger.info("重建完成：%d 份文档", len(stats_list))
        return {"ok": True, "stats": [s.__dict__ for s in stats_list]}

    return app


app = None       # 延迟到 uvicorn 启动时构造，避免导入即加载模型


def _get_app() -> FastAPI:
    """工厂入口（uvicorn --factory 调用）。构造一次并复用，避免重复预热。"""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(_get_app(), host="0.0.0.0", port=8000)
