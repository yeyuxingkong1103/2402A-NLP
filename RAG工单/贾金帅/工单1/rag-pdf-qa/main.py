"""
RAG 招股说明书问答系统 —— 服务入口
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
    python main.py                 # 默认 0.0.0.0:8010
    APP_PORT=8011 python main.py   # 换端口

约定：
  * `/`           保留为 JSON 根路由（探针/脚本依赖，**不要**被静态前端覆盖）
  * `/ui`         零构建静态问答界面
  * `/api/*`      业务接口，统一 {code, message, data, trace_id} 信封
  * `/health` `/ready` `/config`  系统探针，返回裸对象（不套信封）
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from src import config, llm, rag  # noqa: E402
from src.index_store import KnowledgeBase  # noqa: E402

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rag-pdf-qa")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动即预热（见 warm_up 的说明），关闭时不做特殊处理。"""
    logger.info("启动 %s | 工单编号：%s", config.APP_NAME, config.WORK_ORDER_NO)
    warm_up()
    yield


app = FastAPI(
    title="招股说明书 RAG 问答系统",
    description=f"工单编号：{config.WORK_ORDER_NO}",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TS = time.time()


# ------------------------------------------------------------------ 信封


def ok(data, message: str = "OK", trace_id: str | None = None):
    return {"code": "OK", "message": message, "data": data, "trace_id": trace_id or uuid.uuid4().hex[:12]}


def fail(code: str, message: str, trace_id: str | None = None):
    return {"code": code, "message": message, "data": None, "trace_id": trace_id or uuid.uuid4().hex[:12]}


# ------------------------------------------------------------------ 探针


@app.get("/")
def root():
    """保留的 JSON 根路由。"""
    return {
        "service": config.APP_NAME,
        "work_order_no": config.WORK_ORDER_NO,
        "console": "/ui",
        "docs": "/docs",
        "api": ["/api/ask", "/api/ask/stream", "/api/compare", "/api/kb/status", "/api/questions"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TS, 1)}


@app.get("/config")
def get_config():
    """返回非敏感配置（绝不回传 API Key）。"""
    return config.as_public_dict()


@app.get("/ready")
def ready():
    """
    就绪探针。

    注意：这里做的是**运行时依赖可导入性 + 索引真能加载**的检查，
    不是「目录存在就算 ok」——只查路径的探针会给出「全绿但功能全挂」的假信号。
    """
    from importlib.util import find_spec

    components: dict = {}

    # 向量模型依赖
    if find_spec("sentence_transformers") is None:
        components["embedding"] = {
            "status": "unconfigured",
            "reason": "未安装 sentence-transformers，向量化不可用",
        }
    elif not Path(config.EMBEDDING_MODEL_PATH).is_dir():
        components["embedding"] = {
            "status": "unconfigured",
            "reason": f"模型目录不存在：{config.EMBEDDING_MODEL_PATH}",
        }
    else:
        components["embedding"] = {"status": "ok", "model": Path(config.EMBEDDING_MODEL_PATH).name}

    # 索引
    try:
        kb = KnowledgeBase.get()
        components["index"] = {"status": "ok", "chunks": len(kb.chunks)}
    except Exception as exc:  # noqa: BLE001
        components["index"] = {"status": "down", "reason": str(exc)[:200]}

    # 大模型：真发一次最小请求，而不是只看 Key 存不存在
    components["llm"] = llm.health_check()

    overall = "ok"
    if components["index"]["status"] != "ok" or components["llm"]["status"] != "ok":
        overall = "degraded"
    if components["embedding"]["status"] != "ok":
        overall = "degraded"
    return {"status": overall, "components": components}


# ------------------------------------------------------------------ 知识库


@app.get("/api/kb/status")
def kb_status():
    try:
        kb = KnowledgeBase.get()
        return ok(kb.stats())
    except Exception as exc:  # noqa: BLE001
        return ok({"ready": False, "reason": str(exc)})


@app.get("/api/questions")
def questions():
    """工单验收用的 10 个问题。"""
    return ok(load_ticket_questions())


def load_ticket_questions() -> list[dict]:
    path = config.EVAL_DIR / "ticket_questions.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


# ------------------------------------------------------------------ 问答


@app.post("/api/ask")
async def api_ask(payload: dict):
    question = (payload or {}).get("question", "").strip()
    if not question:
        return fail("BAD_REQUEST", "question 不能为空")
    top_k = int(payload.get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    try:
        ans = rag.answer(question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("问答失败")
        return fail("INTERNAL_ERROR", f"问答失败：{exc}")
    return ok(ans.to_dict())


@app.post("/api/compare")
async def api_compare(payload: dict):
    """RAG 结果 vs 纯 LLM 结果。工单明确要求的对比分析。"""
    question = (payload or {}).get("question", "").strip()
    if not question:
        return fail("BAD_REQUEST", "question 不能为空")
    top_k = int(payload.get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    try:
        rag_ans = rag.answer(question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG 分支失败")
        return fail("INTERNAL_ERROR", f"RAG 分支失败：{exc}")
    llm_ans = rag.answer_llm_only(question)
    return ok({"question": question, "rag": rag_ans.to_dict(), "llm_only": llm_ans.to_dict()})


@app.post("/api/ask/stream")
async def api_ask_stream(payload: dict):
    """
    SSE 流式问答。

    帧体统一用 JSON（`data: {"type":...,"data":...}`）：
    裸文本帧里的换行会破坏 SSE 帧边界，把长回答截断成第一段。
    """
    question = (payload or {}).get("question", "").strip()
    top_k = int((payload or {}).get("top_k") or config.RETRIEVAL_FINAL_TOP_K)

    def gen():
        if not question:
            yield f"data: {json.dumps({'type': 'error', 'data': 'question 不能为空'}, ensure_ascii=False)}\n\n"
            return
        try:
            for event, payload_data in rag.answer_stream(question, top_k=top_k):
                frame = json.dumps({"type": event, "data": payload_data}, ensure_ascii=False)
                yield f"data: {frame}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式问答异常")
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)[:300]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ------------------------------------------------------------------ 上传


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """上传 PDF 并重建索引（同步执行，前端用计时器显示已耗时）。"""
    name = file.filename or "upload.pdf"
    if not name.lower().endswith(".pdf"):
        return fail("BAD_REQUEST", "只支持 PDF 文件")

    config.ensure_dirs()
    dest = config.UPLOAD_DIR / name
    raw = await file.read()
    if len(raw) > config.UPLOAD_MAX_MB * 1024 * 1024:
        return fail("BAD_REQUEST", f"文件超过 {config.UPLOAD_MAX_MB}MB 上限")
    dest.write_bytes(raw)

    try:
        result = rebuild_index(dest)
    except Exception as exc:  # noqa: BLE001
        logger.exception("索引重建失败")
        return fail("INDEX_ERROR", f"解析或向量化失败：{exc}")
    return ok(result, message="索引已重建")


def rebuild_index(pdf_path: Path) -> dict:
    """解析 → 分块 → 向量化 → 落盘。同时被上传接口与 scripts/build_index.py 复用。"""
    from src.chunker import chunk_document
    from src.embedder import embed_texts
    from src.index_store import save_index
    from src.pdf_parser import parse_pdf

    t0 = time.perf_counter()
    parsed = parse_pdf(pdf_path, with_tables=True)
    chunks = chunk_document(parsed)
    vecs = embed_texts([c.text for c in chunks])
    meta = save_index(chunks, vecs, pdf_path, config.EMBEDDING_MODEL_PATH)
    KnowledgeBase.reset()
    return {**meta, "elapsed_seconds": round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------ 静态前端


def _mount_console() -> None:
    """
    挂在 `/ui`，**不要**挂在 `/`：挂 `/` 会静默覆盖既有 JSON 根路由，
    而部署脚本、探针很可能依赖那个约定。
    """
    static_dir = config.STATIC_DIR
    if not static_dir.is_dir():
        logger.warning("静态目录不存在，前端未挂载：%s", static_dir)
        return
    try:
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover
        logger.warning("fastapi.staticfiles 不可用，前端未挂载：%s", exc)
        return
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="console")


_mount_console()


# ------------------------------------------------------------------ 预热


def warm_up() -> None:
    """
    启动即预热，把一次性开销从「第一个用户请求」挪到「服务启动」。

    不预热的话，首个提问要额外承担：
      索引加载 + BM25 分词建表（~2s）+ 向量模型加载（~3s）+ LLM 连接握手（~10s）
    实测首问会到 39 秒，远超过工单的 3 秒要求。
    预热失败只告警、不阻止启动 —— 服务本身要能起来，索引问题应该体现在 /ready 里。
    """
    t0 = time.perf_counter()
    try:
        kb = KnowledgeBase.get()
        _ = kb.bm25                       # 触发 jieba 分词 + BM25 建表
        from src.embedder import embed_query

        embed_query("预热")               # 触发向量模型加载
        logger.info("预热完成：索引 %d 块，耗时 %.1fs", len(kb.chunks), time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("索引/向量预热失败（%s），请检查是否已建索引", exc)

    try:
        t1 = time.perf_counter()
        llm.chat([{"role": "user", "content": "ping"}], max_tokens=4)
        logger.info("大模型连接预热完成，耗时 %.1fs", time.perf_counter() - t1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("大模型预热失败（%s），请检查 LLM_API_KEY", exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.APP_HOST, port=config.APP_PORT, log_level=config.LOG_LEVEL.lower())
