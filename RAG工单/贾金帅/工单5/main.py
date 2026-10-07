"""
RAG 招股说明书问答系统 —— 服务入口
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
工单编号：人工智能NLP-RAG-Query理解优化任务

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
from src.retriever import build_context, retrieve  # noqa: E402
from src.session import get_session, store as session_store_func  # noqa: E402

session_store = session_store_func()

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rag-pdf-qa")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动即预热（见 warm_up 的说明），关闭时不做特殊处理。"""
    logger.info("启动 %s | 工单：%s / %s / %s / %s", config.APP_NAME,
                config.WORK_ORDER_NO, config.WORK_ORDER_NO_OPT, config.WORK_ORDER_NO_TABLE,
                config.WORK_ORDER_NO_IMAGE)
    warm_up()
    yield


app = FastAPI(
    title="招股说明书 RAG 问答系统",
    description=f"工单编号：{config.WORK_ORDER_NO_IMAGE}",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TS = time.time()


# ------------------------------------------------------------------ 多轮会话（工单05）


_MODES = ("off", "concat", "rule", "rule+llm")


def _resolve_session(payload: dict):
    """
    从请求体里取出会话与消解模式。

    两个约定：
      * **不传 session_id 就完全是单轮** —— 工单1~4 的所有对外行为必须保持不变，
        多轮只能是一个「显式开启」的能力，不能悄悄改变既有接口的语义。
      * `mode` 传了非法值不报错、退回默认配置：这是演示与压测入口，
        为了一个拼错的枚举值让整条问答链路失败，代价不对等。
    """
    payload = payload or {}
    sid = (payload.get("session_id") or "").strip()
    mode = (payload.get("mode") or "").strip().lower()
    if not sid:
        return None, None
    if mode not in _MODES:
        mode = None
    return get_session(sid), mode


@app.get("/api/session/stats")
def api_session_stats():
    """
    会话存储的运行指标 —— 对应验收标准「资源消耗合理、高并发稳定」。

    会话是**进程内有上限**的（见 src/session.py 的说明），
    这里把「当前多少会话 / 上限多少 / 每会话最多几轮」暴露出来，
    压测时可以直接验证它不会无限增长。

    ⚠️ 必须声明在 `/api/session/{session_id}` 之前：FastAPI 按声明顺序匹配，
    反过来的话 `/api/session/stats` 会被路径参数吃掉，返回「会话 stats 不存在」
    （实测过，确实是这个现象）。
    """
    return ok(session_store.stats())


@app.get("/api/session/{session_id}")
def api_session_get(session_id: str):
    """查看会话状态（回看系统把每一轮理解成了什么）。"""
    s = session_store.peek(session_id)
    if s is None:
        return ok({"exists": False, "session_id": session_id})
    return ok({"exists": True, **_session_view(s)})


@app.post("/api/session/reset")
async def api_session_reset(payload: dict):
    """清空某个会话的上下文（保留 session_id）。"""
    sid = ((payload or {}).get("session_id") or "").strip()
    if not sid:
        return fail("BAD_REQUEST", "缺少 session_id")
    hit = session_store.reset(sid)
    return ok({"session_id": sid, "reset": hit})


@app.delete("/api/session/{session_id}")
def api_session_delete(session_id: str):
    """删除会话。"""
    return ok({"session_id": session_id, "deleted": session_store.drop(session_id)})


def _session_view(s) -> dict | None:
    """会话的状态快照，回给前端做「上下文」面板。"""
    if s is None:
        return None
    d = s.to_dict(include_answers=False)
    d["mode"] = config.MULTITURN_MODE
    d["max_turns"] = config.MULTITURN_MAX_TURNS
    return d


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
        "work_order_no_opt": config.WORK_ORDER_NO_OPT,
        "work_order_no_table": config.WORK_ORDER_NO_TABLE,
        "work_order_no_image": config.WORK_ORDER_NO_IMAGE,
        "work_order_no_qu": config.WORK_ORDER_NO_QU,
        # 注意：索引元信息里的 work_order_nos 是**建索引那一刻**的快照，
        # 后加的工单编号不会自动出现在其中（重建索引代价太大），
        # 所以这里再给一份「代码里现在有哪些工单」的权威列表。
        "work_order_nos": list(config.WORK_ORDER_NOS),
        "corpus": [d["name"] for d in config.DOCS],
        "console": "/ui",
        "docs": "/docs",
        "api": [
            "/api/ask",
            "/api/ask/stream",
            "/api/session/{session_id}",
            "/api/session/reset",
            "/api/session/stats",
            "/api/search",
            "/api/compare",
            "/api/compare-opt",
            "/api/compare-table",
            "/api/compare-image",
            "/api/kb/status",
            "/api/questions",
        ],
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
    """知识库状态。工单03/04 起会同时报告两份对照索引是否就绪。"""
    try:
        kb = KnowledgeBase.get()
        data = kb.stats()
        data["image_chunks"] = sum(1 for c in kb.chunks if c.get("type") == "image")
        try:
            nb = KnowledgeBase.get_notable()
            data["no_table_index"] = {"ready": True, "chunks": len(nb.chunks),
                                      "table_chunks": 0}
        except Exception as exc:  # noqa: BLE001
            data["no_table_index"] = {"ready": False, "reason": str(exc)}
        try:
            ib = KnowledgeBase.get_noimage()
            data["no_image_index"] = {"ready": True, "chunks": len(ib.chunks),
                                      "image_chunks": 0}
        except Exception as exc:  # noqa: BLE001
            data["no_image_index"] = {"ready": False, "reason": str(exc)}
        data["clip"] = _clip_status()
        return ok(data)
    except Exception as exc:  # noqa: BLE001
        return ok({"ready": False, "reason": str(exc)})


def _image_url(rel: str) -> str:
    """把块里存的图路径（相对工程根目录，如 `data/images/liyuan/p39_f1.png`）
    转成前端可取的 URL（`/images/liyuan/p39_f1.png`）。

    为什么要转：`image` 字段是**落盘相对路径**，直接丢给浏览器会被当成
    相对 `/ui/` 的路径而 404；统一在这里转一次，前端只认 `/images/...`。

    工单4 排查记录：索引里存的路径可能是 `data\\images\\xingtu\\p99_f1.png`
    这种**反斜杠**写法（JSON 落盘时保留了 Windows 分隔符），而 `IMAGE_DIR`
    是绝对路径 —— 直接 `Path(rel).relative_to(IMAGE_DIR)` 会抛 ValueError、
    静默返回空串，表现为「检索命中了图，但界面上不显示图」。
    因此这里先把分隔符统一成正斜杠，再按「相对工程根目录」补全，
    最后才对齐到 `IMAGE_DIR`。两种写法都能取到图。
    """
    if not rel:
        return ""
    p = Path(str(rel).replace("\\", "/"))
    if not p.is_absolute():
        p = config.ROOT_DIR / p
    try:
        return "/images/" + p.resolve().relative_to(config.IMAGE_DIR.resolve()).as_posix()
    except ValueError:
        return ""


def _clip_status() -> dict:
    """CLIP 通道状态：向量矩阵在不在、模型目录在不在。只报事实，不做加载。"""
    try:
        from src import clip_encoder

        enc = clip_encoder.ClipEncoder.instance()
        vecs, figures = clip_encoder.load_clip_index()
        return {
            "enabled": config.CLIP_ENABLED,
            "model_path": enc.model_path,
            "model_present": enc.available,
            "vectors": 0 if vecs is None else int(vecs.shape[0]),
            "dim": 0 if vecs is None else int(vecs.shape[1]),
            "figures": len(figures),
            "load_error": enc.load_error,
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": config.CLIP_ENABLED, "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/questions")
def questions():
    """工单验收用的题目：工单4 = 2 道图像题（力源）+ 工单3 的 14 题，共 16 题。"""
    return ok(load_ticket_questions())


def load_ticket_questions() -> list[dict]:
    """
    读取验收题目。工单04 起评测集是 data/eval/questions_wot4.json（16 题），
    文件不存在时逐级退回工单3、工单1/2 的题集，保证接口永远有东西可返回。
    """
    for name in ("questions_wot4.json", "questions_wot3.json",
                 "questions_wot1_2.json", "ticket_questions.json"):
        path = config.EVAL_DIR / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return []


# ------------------------------------------------------------------ 问答


def _clean_question(payload: dict) -> tuple[str | None, str | None]:
    """
    输入规整 + 校验（工单2「容错机制：能够处理常见的异常情况，如用户输入错误」）。

    返回 (question, error)。error 非空时 question 为 None。

    三种常见异常在这里一次性挡掉，不让它们流进检索/生成：
      * payload 不是对象、question 不是字符串（前端传错类型）
      * 空白输入
      * 超长输入（截断而非报错 —— 用户的诉求仍然要满足，只是不能让它撑爆上下文预算）
    """
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"
    raw = payload.get("question", "")
    if not isinstance(raw, str):
        return None, "question 必须是字符串"
    q = raw.strip()
    if not q:
        return None, "question 不能为空"
    if len(q) > config.MAX_QUESTION_CHARS:
        logger.info("问题过长（%d 字），已截断到 %d 字", len(q), config.MAX_QUESTION_CHARS)
        q = q[: config.MAX_QUESTION_CHARS]
    return q, None


def _norm_fact(s: str) -> str:
    """关键事实比对的归一化：NFKC（全角→半角、「−」→「-」）+ 抹掉所有空白。

    工单04 起必须这么做：图内文字来自多模态模型的转写，空格口径与 PDF 文字层
    不一致（「IC卡」vs「IC 卡」、「2008年中国」vs「2008 年中国」），
    按原字符比对会出现「明明答对了却判成漏」的假阴性。
    """
    import unicodedata

    return "".join(unicodedata.normalize("NFKC", s or "").split())


def _fact_hits(question: str, contexts: list[str]) -> dict | None:
    """
    若问题正好是工单验收题之一，顺带算一下「关键事实命中情况」。

    这是给演示用的：跑一遍优化前后对比，能直接看到「优化前漏了哪些数字」。
    非工单问题返回 None，不参与任何评分。
    """
    for item in load_ticket_questions():
        # 工单4 修正：这里原先用 question.strip() 精确比对，全角/空格变体就对不上，
        # 表现为「明明问的是验收题，命中清单却是 null」。改用与事实匹配同一套归一化。
        if _norm_fact(item.get("question", "")) != _norm_fact(question):
            continue
        must = item.get("must_have") or []
        if not must:
            return None
        joined = _norm_fact("\n".join(contexts))
        hit = [k for k in must if _norm_fact(k) in joined]
        return {
            "id": item.get("id"),
            "must_have": must,
            "hit": hit,
            "missed": [k for k in must if k not in hit],
            "rate": round(len(hit) / len(must), 4),
            # 「层级数量」这类弱判定（正文里到处是 4 和 6）单独列，不并进 rate
            "counts_expected": item.get("must_have_counts") or [],
            "counts_hit": [c for c in (item.get("must_have_counts") or [])
                           if _norm_fact(c) in joined],
            "image_only": bool(item.get("image_only")),
        }
    return None


@app.post("/api/ask")
async def api_ask(payload: dict):
    """
    非流式问答。

    工单5 扩展：可选 `session_id` / `mode`
      * 传 `session_id` → 走多轮：本轮先做指代/省略消解，再检索作答，
        并把本轮的「主体 / 话题 / 消解后问句」回写进会话供下一轮使用。
      * 不传 → 行为与工单1~4 完全一致（单轮），方便做前后对比。
    """
    question, err = _clean_question(payload)
    if err:
        return fail("BAD_REQUEST", err)
    top_k = int((payload or {}).get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    session, mode = _resolve_session(payload)
    try:
        ans = rag.answer(question, top_k=top_k, session=session, mode=mode)
    except Exception as exc:  # noqa: BLE001
        logger.exception("问答失败")
        return fail("INTERNAL_ERROR", f"问答失败：{exc}")
    data = ans.to_dict()
    data["session"] = _session_view(session)
    return ok(data)


@app.post("/api/search")
async def api_search(payload: dict):
    """
    **只检索、不生成**（工单2 稳定性验证用）。

    存在的理由：`/api/ask` 的耗时绝大部分是上游大模型的往返，用它做并发压测，
    测到的其实是「模型服务账号级并发上限」，而不是本系统的并发能力。
    这个接口走的是**纯本地路径**（规则归一化 → 向量 + BM25 → 闸门），
    不碰任何外部服务，因此可以干净地回答「系统侧到底能不能并发」。

    同时它也是个好用的排障入口：想看某个问题实际召回了什么，直接打这个接口。
    """
    question, err = _clean_question(payload)
    if err:
        return fail("BAD_REQUEST", err)
    top_k = int((payload or {}).get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    use_clip = (payload or {}).get("clip")
    use_clip = True if use_clip is None else bool(use_clip)
    try:
        r = retrieve(question, top_k=top_k, use_clip=use_clip)
    except Exception as exc:  # noqa: BLE001
        logger.exception("检索失败")
        return fail("INTERNAL_ERROR", f"检索失败：{exc}")
    return ok({
        "question": question,
        "retrieval_query": r.trace.get("effective_query") or r.used_query,
        "items": [
            {
                "rank": i,
                "page": it.page,
                "page_end": it.page_end,
                "section": it.section,
                "type": it.type,
                "doc_key": it.doc_key,
                "doc": it.doc,
                "evidence": round(it.evidence, 4),
                "score": round(it.score, 4),
                "clip_sim": round(it.clip_sim, 4),
                # 图像块回传裁剪图路径与图类型，前端据此直接回显原图
                "image": _image_url(it.image),
                "fig_type": it.fig_type,
                "caption": it.caption,
                "preview": it.text[:120],
            }
            for i, it in enumerate(r.items, 1)
        ],
        "n_image_chunks": sum(1 for it in r.items if it.type == "image"),
        "steps": r.trace.get("steps") or [],
        "latency_ms": r.trace.get("total_ms", 0.0),
        "gated": r.gated,
    })


@app.post("/api/compare")
async def api_compare(payload: dict):
    """RAG 结果 vs 纯 LLM 结果。工单1 明确要求的对比分析。"""
    question, err = _clean_question(payload)
    if err:
        return fail("BAD_REQUEST", err)
    top_k = int((payload or {}).get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    try:
        rag_ans = rag.answer(question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG 分支失败")
        return fail("INTERNAL_ERROR", f"RAG 分支失败：{exc}")
    llm_ans = rag.answer_llm_only(question)
    return ok({"question": question, "rag": rag_ans.to_dict(), "llm_only": llm_ans.to_dict()})


@app.post("/api/compare-opt")
async def api_compare_opt(payload: dict):
    """
    **工单2 核心接口**：优化后（本系统） vs 优化前（朴素 RAG） vs 纯 LLM。

    三条链路回答同一个问题，并（若命中验收题）给出关键事实命中情况，
    让「优化前后的检索精确度变化」在界面上一眼可见。
    """
    question, err = _clean_question(payload)
    if err:
        return fail("BAD_REQUEST", err)
    body = payload or {}
    top_k = int(body.get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    include_llm = bool(body.get("include_llm"))

    try:
        opt = rag.answer(question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("优化后分支失败")
        return fail("INTERNAL_ERROR", f"优化后分支失败：{exc}")

    opt_contexts = [c["text"] for c in opt.citations]

    try:
        from src.baseline import answer_naive

        naive = answer_naive(question, top_k=top_k)
    except FileNotFoundError as exc:
        # 基线索引没建 → 只回优化后那份，并如实说明，不让整个接口失败
        logger.warning("朴素基线不可用：%s", exc)
        return ok({
            "question": question,
            "optimized": opt.to_dict(),
            "naive": None,
            "naive_unavailable_reason": str(exc),
            "fact_hits": {
                "optimized": _fact_hits(question, opt_contexts),
                "naive": None,
            },
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("朴素基线分支失败")
        return ok({
            "question": question,
            "optimized": opt.to_dict(),
            "naive": None,
            "naive_unavailable_reason": f"朴素基线执行失败：{exc}",
            "fact_hits": {"optimized": _fact_hits(question, opt_contexts), "naive": None},
        })

    data = {
        "question": question,
        "optimized": opt.to_dict(),
        "naive": naive,
        "fact_hits": {
            "optimized": _fact_hits(question, opt_contexts),
            "naive": _fact_hits(question, naive.get("contexts") or []),
        },
    }
    if include_llm:
        data["llm_only"] = rag.answer_llm_only(question).to_dict()
    return ok(data)


@app.post("/api/compare-table")
async def api_compare_table(payload: dict):
    """
    **工单03 核心接口**：表格「不结构化（优化前）」 vs 「结构化（本工单实现）」。

    两条链路**只差一个变量** —— 喂给检索与生成的那份分块来自哪一份索引：
      * /api/compare-opt  对比的是工单2 的朴素 RAG（定长分块 + 纯向量），跨了多个变量；
      * 本接口           对比的是同一套解析/分块/检索，只有「表格有没有被结构化」不同。

    工单3 的验收要求「显示检索到的答案及检索精确度」，所以这里同时给出：
      * 两条链路**各自召回的依据片段**（含页码与来源文档）
      * 两条链路**各自的答案**（同一段 prompt、同一个模型）
      * 若问题命中验收题目，附上关键事实命中清单（must_have）
    """
    question, err = _clean_question(payload)
    if err:
        return fail("BAD_REQUEST", err)
    body = payload or {}
    top_k = int(body.get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    include_llm = bool(body.get("include_llm"))

    try:
        # 工单3 的对比只解释「表格有没有结构化」这一个变量，
        # 因此这里把 CLIP 通道关掉 —— 图像召回属于工单4，不该混进工单3 的结论。
        opt = rag.answer(question, top_k=top_k, use_clip=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("表格结构化分支失败")
        return fail("INTERNAL_ERROR", f"表格结构化分支失败：{exc}")

    try:
        plain = _answer_with_kb(question, KnowledgeBase.get_notable(), top_k=top_k,
                                use_clip=False)
    except FileNotFoundError as exc:
        logger.warning("对照索引不可用：%s", exc)
        return ok({
            "question": question,
            "table": opt.to_dict(),
            "no_table": None,
            "no_table_unavailable_reason": str(exc),
            "fact_hits": {"table": _fact_hits(question, [c["text"] for c in opt.citations]),
                          "no_table": None},
        })

    data = {
        "question": question,
        "table": opt.to_dict(),
        "no_table": plain,
        "fact_hits": {
            "table": _fact_hits(question, [c["text"] for c in opt.citations]),
            "no_table": _fact_hits(question, plain.get("contexts") or []),
        },
    }
    if include_llm:
        data["llm_only"] = rag.answer_llm_only(question).to_dict()
    return ok(data)


def _answer_with_kb(question: str, kb: KnowledgeBase, top_k: int,
                    use_clip: bool = True) -> dict:
    """
    用指定知识库跑一遍「检索 → 组装上下文 → 生成」，返回与 rag.Answer 对齐的 dict。

    存在的理由：`rag.answer()` 走的是全局单例（主线索引），
    而工单03 需要让**同一段 prompt、同一个模型**跑在另一份分块上。
    复用同一个 system prompt（`rag._RAG_SYSTEM`）是刻意的 ——
    生成侧的任何差异都必须来自「喂进去的资料」，不能来自提示词。
    """
    t0 = time.perf_counter()
    r = retrieve(question, top_k=top_k, kb=kb, use_clip=use_clip)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    items = list(r.items)
    contexts = [it.text for it in items]
    ctx_text = build_context(items, max_chars=4800)

    if not items:
        return {
            "question": question, "answer": rag.NO_EVIDENCE_TEXT, "mode": "no_evidence",
            "citations": [], "contexts": [], "context_text": "",
            "images": [],
            "timing": {"retrieval_ms": round(retrieval_ms, 2),
                       "total_ms": round((time.perf_counter() - t0) * 1000, 2)},
        }

    t = time.perf_counter()
    text = llm.chat(
        [
            {"role": "system", "content": rag._RAG_SYSTEM},
            {"role": "user",
             "content": f"【资料片段】\n{ctx_text}\n\n【问题】\n{question}\n\n请依据上述资料片段回答。"},
        ]
    )
    gen_ms = (time.perf_counter() - t) * 1000
    return {
        "question": question,
        "answer": text.strip(),
        "mode": "rag",
        "citations": rag.build_citations(items),
        "contexts": contexts,
        "context_text": ctx_text,
        # 召回里的图像块：前端直接渲染裁剪出的图，演示时能"看到答案出自哪张图"
        "images": [
            {"rank": i, "page": it.page, "image": _image_url(it.image),
             "fig_type": it.fig_type, "caption": it.caption, "doc_key": it.doc_key,
             "evidence": round(it.evidence, 4), "clip_sim": round(it.clip_sim, 4)}
            for i, it in enumerate(items, 1) if it.type == "image"
        ],
        "steps": r.trace.get("steps") or [],
        "timing": {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        },
    }


@app.post("/api/compare-image")
async def api_compare_image(payload: dict):
    """
    **工单04 核心接口**：图像「不解析（优化前）」 vs 「多模态解析（本工单实现）」。

    与 /api/compare-table 同一套设计：两条链路**只差一个变量** ——
    喂给检索与生成的分块来自哪一份索引。解析、分块、检索、prompt、模型全部相同。

    工单验收要求「显示检索到的答案及检索精确度」，因此这里同时给出：
      * 两条链路**各自召回的依据片段**（含页码、类型、依据分）
      * 两条链路**各自生成的答案**（同一段 prompt、同一个模型）
      * 命中验收题时附上关键事实命中清单（must_have），图像题还额外给出
        «层级数量» 这类弱判定（must_have_counts），但**单独列出、不并入覆盖率**
      * 召回中的图像块回传裁剪图路径，界面直接显示原图

    可选 use_clip（默认 True）：第三条链路「多模态解析 + CLIP 跨模态召回」。
    """
    question, err = _clean_question(payload)
    if err:
        return fail("BAD_REQUEST", err)
    body = payload or {}
    top_k = int(body.get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    include_clip = body.get("include_clip")
    include_clip = True if include_clip is None else bool(include_clip)

    try:
        opt = _answer_with_kb(question, KnowledgeBase.get(), top_k, use_clip=include_clip)
    except Exception as exc:  # noqa: BLE001
        logger.exception("图像解析分支失败")
        return fail("INTERNAL_ERROR", f"图像解析分支失败：{exc}")

    try:
        plain = _answer_with_kb(question, KnowledgeBase.get_noimage(), top_k, use_clip=False)
    except FileNotFoundError as exc:
        logger.warning("图像对照索引不可用：%s", exc)
        return ok({
            "question": question,
            "image": opt,
            "no_image": None,
            "no_image_unavailable_reason": str(exc),
            "fact_hits": {"image": _fact_hits(question, opt.get("contexts") or []),
                          "no_image": None},
        })

    data = {
        "question": question,
        "image": opt,
        "no_image": plain,
        "fact_hits": {
            "image": _fact_hits(question, opt.get("contexts") or []),
            "no_image": _fact_hits(question, plain.get("contexts") or []),
        },
    }
    return ok(data)


@app.post("/api/ask/stream")
async def api_ask_stream(payload: dict):
    """
    SSE 流式问答。

    帧体统一用 JSON（`data: {"type":...,"data":...}`）：
    裸文本帧里的换行会破坏 SSE 帧边界，把长回答截断成第一段。
    """
    question, err = _clean_question(payload)
    top_k = int((payload or {}).get("top_k") or config.RETRIEVAL_FINAL_TOP_K)
    session, mode = _resolve_session(payload)

    def gen():
        if err:
            yield f"data: {json.dumps({'type': 'error', 'data': err}, ensure_ascii=False)}\n\n"
            return
        try:
            for event, payload_data in rag.answer_stream(question, top_k=top_k,
                                                         session=session, mode=mode):
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
    # 裁剪出的图单独挂一个前缀（工单04）：演示时要能"看到答案出自哪张图"。
    # 不能塞进 /ui 下面 —— 图片落在 data/images/，与前端源码目录不是同一个地方，
    # 复制过去会在重建图时两头不同步。
    if config.IMAGE_DIR.is_dir():
        app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")


_mount_console()


# ------------------------------------------------------------------ 预热


def warm_up() -> None:
    """
    启动即预热，把一次性开销从「第一个用户请求」挪到「服务启动」。

    不预热的话，首个提问要额外承担：
      索引加载 + BM25 分词建表（~2s）+ 语料级无区分度短语表（~0.3s，工单2 新增）
      + 向量模型加载（~3s）+ LLM 连接握手（~10s）
    实测首问会到 39 秒，远超过工单的 3 秒要求。
    预热失败只告警、不阻止启动 —— 服务本身要能起来，索引问题应该体现在 /ready 里。
    """
    t0 = time.perf_counter()
    try:
        kb = KnowledgeBase.get()
        _ = kb.bm25                       # 触发 jieba 分词 + BM25 建表
        phrases = kb.ubiquitous_phrases   # 触发语料级无区分度短语表（工单2 新增）
        from src.embedder import embed_query

        embed_query("预热")               # 触发向量模型加载
        logger.info("预热完成：索引 %d 块，无区分度短语 %d 条，耗时 %.1fs",
                    len(kb.chunks), len(phrases), time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("索引/向量预热失败（%s），请检查是否已建索引", exc)

    # 朴素基线索引是可选对照，预热失败不影响主线
    try:
        from src.baseline import get_naive_index

        _ = get_naive_index()
        logger.info("朴素基线索引已就绪（优化前后对比用）")
    except Exception as exc:  # noqa: BLE001
        logger.info("朴素基线索引未就绪（%s），/api/compare-opt 将只返回优化后结果", exc)

    try:
        t1 = time.perf_counter()
        llm.chat([{"role": "user", "content": "ping"}], max_tokens=4)
        logger.info("大模型连接预热完成，耗时 %.1fs", time.perf_counter() - t1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("大模型预热失败（%s），请检查 LLM_API_KEY", exc)

    # 工单04：CLIP 跨模态通道预热。
    # **只在向量矩阵已经存在时才加载模型** —— 没有向量的情况下加载 CLIP 纯属浪费
    # （几百 MB 权重的 IO 会拖慢启动，而且是永远用不上的那份开销）。
    if config.CLIP_ENABLED:
        try:
            from src import clip_encoder

            vecs, _figs = clip_encoder.load_clip_index()
            if vecs is None:
                logger.info("CLIP 通道未就绪：无图向量（先跑 scripts/parse_images.py --clip-only）")
            else:
                t1 = time.perf_counter()
                enc = clip_encoder.ClipEncoder.instance()
                if enc.load():
                    _ = enc.encode_texts(["预热"])
                    logger.info("CLIP 通道预热完成（%d 张图向量），耗时 %.1fs",
                                vecs.shape[0], time.perf_counter() - t1)
                else:
                    logger.warning("CLIP 模型加载失败：%s（检索将自动跳过该通道）",
                                   enc.load_error)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLIP 预热失败（%s），检索会自动降级", exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.APP_HOST, port=config.APP_PORT, log_level=config.LOG_LEVEL.lower())
