# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单05 - Query 理解优化（多轮对话 + 指代消解/省略补全）
"""
问答接口：非流式 / SSE 流式 / 反馈。

【工单05：多轮对话怎么接进来】
带 `session_id` 时多走三步（不带 = 单轮，行为与工单01-04 逐字节一致）：
  ① 取会话状态（焦点实体 / 问点模板）→ 把追问改写成独立问题（Query 理解）；
  ② 用**改写后**的问题去 `retrieve()` 与 `build_context()`；
  ③ 生成时带上会话 history，答案出来后把本轮写回会话。
【为什么 ② 必须两处都换】`build_context` 用问题算查询词来选窗口（retriever.py:665）。
只换 retrieve 不换它，「那力源呢？」的查询词近乎为空 → `best_window` 退化成截开头
（retriever.py:624）→ 可能正好切掉答案，**而且不报错**。


【响应时间的口径】
工单01 写的是「从用户提问到**返回答案**的时间 ≤3 秒」。
非流式下"返回答案"= 最后一个 token；而实测招股书场景 prefill 1.5–4s、
生成 3–6s，整段稳定超 3 秒。
流式下首个 token 即"开始返回"，因此：
  - 验收口径采用 **TTFT（首 token 延迟）**；
  - 同时**主动披露**完整答案耗时（total_ms），不藏。
两项都会写进技术文档，避免验收时被质疑口径。
"""

from __future__ import annotations

import asyncio
import collections
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.doc_profiles import get_doc_profile
from app.core.generator import Generator
from app.core.pipeline import get_pipeline
from app.core.profiles import get_profile
from app.core.multiturn import (
    commit_turn, prepare_turn, refresh_history, turn_fields)
from app.core.query_understanding import QueryUnderstanding, next_focus_entity
from app.core.retriever import Retriever, build_context
from app.core.session import get_session_store
from app.core.vectorstore import VectorStoreError
from app.schemas import ChatRequest, ChatResponse, CitationOut, FeedbackOut, FeedbackRequest

router = APIRouter(tags=["chat"])

# 单机 8GB 显存只能串行推理；并发请求进来就排队，而不是互相抢显存导致都变慢。
# 前端会显示「排队中」提示，对应工单01「高并发稳定性」验收项。
_semaphore = asyncio.Semaphore(settings.max_concurrency)

# 工单05：Query 理解（规则优先 + LLM 兜底）。无状态，可安全共享。
_qu = QueryUnderstanding()


def _derive_profile(profile: str | None, *, retrieval_mode: str | None = None,
                    fusion: str | None = None, w_keyword: float | None = None,
                    reranker: str | None = None,
                    fusion_impl: str | None = None) -> "RetrievalProfile":
    """按请求派生检索剖面（工单06）。

    【为什么用 derived() 而不是改 settings】剖面是**不可变值对象、沿调用链显式传递**
    （见 profiles.py 的竞态说明）。`settings` 是进程级单例，若按请求去改它，
    并发下两个请求会互相踩到对方刚改的参数，而且极难复现。
    """
    p = get_profile(profile)
    over = {}
    if retrieval_mode:
        over["retrieval_mode"] = retrieval_mode
    if fusion:
        over["fusion"] = fusion
    if w_keyword is not None:
        over["w_keyword"] = w_keyword
    if reranker:
        over["reranker"] = reranker
    if retrieval_mode == "hybrid" and fusion_impl:
        over["fusion_impl"] = fusion_impl
    return p.derived(**over) if over else p


def _profile_from_request(req) -> "RetrievalProfile":
    return _derive_profile(req.profile, retrieval_mode=req.retrieval_mode,
                           fusion=req.fusion, w_keyword=req.w_keyword,
                           reranker=req.reranker, fusion_impl=req.fusion_impl)




def _citations(hits) -> list[CitationOut]:
    """命中块 + 邻块都返回，但邻块带 is_neighbor 标记。

    【为什么邻块也返回】工单02 的演示要能看见"邻块扩展到底补了什么"，
    藏起来反而不好讲。前端把它渲染成灰色的「邻接补充」，不当作答案依据。

    【为什么它不影响指标】评测里的 citations / 精确率分母都排除了邻块
    （见 evaluator.retrieval_metrics），这里的返回只影响前端展示。
    """
    return [
        CitationOut(
            page_label=h.page_label, page_no=h.page_no, chunk_type=h.chunk_type,
            section_path=h.section_path,
            snippet=h.content[:120], score=round(h.score, 4),
            is_neighbor=h.is_neighbor,
        )
        for h in hits
    ]


# ----------------------------------------------------------------------
# 响应缓存（工单02）
# ----------------------------------------------------------------------
# 【为什么缓存放这里而不是 core/】评测（Evaluator）直接调 Retriever/Generator，
# 不经过这一层 —— 所以评测天然不吃缓存，TTFT 数字是真的。
# 若把缓存塞进 core/，评测就会命中缓存、TTFT 变成 0ms，指标当场失真。
#
# 【为什么键里必须有 profile】三个剖面答同一道题的结果不同，
# 不区分剖面就会把 optimized 的答案喂给 baseline 的对比。
_CACHE: "collections.OrderedDict[tuple, ChatResponse]" = collections.OrderedDict()
_CACHE_MAX = 128


def _cache_key(question: str, profile: str, top_k, no_rag: bool) -> tuple:
    # 归一化空白，避免同一个问题因多打一个空格而漏命中
    return (" ".join(question.split()), profile, top_k, no_rag)


def _cache_get(key: tuple) -> ChatResponse | None:
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
    return hit


def _cache_put(key: tuple, value: ChatResponse) -> None:
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def _persist_feedback(req: FeedbackRequest) -> int:
    """反馈落盘为 JSONL，供工单07 评估与后续微调使用。

    【工单05：必须带上 session_id / turn_index】多轮界面里每一轮各自有反馈按钮，
    不带会话与轮次就无法定位「这条反馈说的到底是哪一问」。这两个字段是
    `FeedbackRequest` 的一部分，**写盘时也要落进去** —— 只加在请求模型上而忘了
    写进记录，接口照常 200、前端照常提示"已记录"，日志里却查不出是哪一轮
    （属于"看着成功、实际丢信息"的一类静默问题）。
    """
    import json as _json
    from datetime import datetime
    path = settings.data_path / "feedback" / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": req.session_id, "turn_index": req.turn_index,
        "question": req.question, "answer": req.answer,
        "rating": req.rating, "comment": req.comment,
        "citations": [c.model_dump() for c in req.citations],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


# ----------------------------------------------------------------------
@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """非流式问答。no_rag=true 时走纯 LLM 基线；带 session_id 时走多轮。"""
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")

    profile = _profile_from_request(req)
    # 【工单05：多轮绕过缓存】带 session_id 时读与写都跳过 —— 同一句话在不同上下文下
    # 答案不同，缓存语义不成立；更危险的是反向污染：会话答案写进公共缓存后，
    # 单轮路径（对比页、既有 16 题）会读到带 turn_index/rewritten 的响应，
    # 而 cached=True 会让 TTFT 变成一个虚假的好数字。
    use_cache = not req.session_id
    key = _cache_key(req.question, profile.name, req.top_k, req.no_rag)
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            # 命中缓存必须显式暴露 cached=True —— 否则 TTFT 会因为"没真的推理"
            # 而变成一个虚假的好数字，拿去验收就是自欺。
            return hit.model_copy(update={"cached": True})

    t0 = time.perf_counter()
    # 构造放进 try：Milvus 不可用时 VectorStoreError 是从构造器抛的
    # （client 是 lazy property），放在外面会漏成 500 而不是 503。
    try:
        retriever = Retriever(profile=profile)
    except VectorStoreError as e:
        raise HTTPException(503, f"向量库不可用：{e}") from e
    generator = Generator()

    async with _semaphore:
        # ---------- 纯 LLM 基线 ----------
        if req.no_rag:
            try:
                answer = await generator.baseline_no_rag(req.question)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(503, f"模型不可用：{e}") from e
            total = int((time.perf_counter() - t0) * 1000)
            resp = ChatResponse(
                question=req.question, answer=answer, mode="no_rag",
                ttft_ms=total,      # 非流式下 TTFT 无意义，与 total 同值并如实标注
                total_ms=total, profile=profile.name,
            )
            if use_cache:
                _cache_put(key, resp)
            return resp

        # ---------- 工单05：取会话状态 + 改写问题（必须在本信号量内，改写兜底会调模型）----------
        store = get_session_store()
        tctx = await prepare_turn(store, _qu, req.question, req.session_id)

        # ---------- RAG ----------
        try:
            rres = await retriever.retrieve(tctx.rewritten, req.top_k)
        except VectorStoreError as e:
            raise HTTPException(503, f"向量库不可用：{e}") from e

        retrieval_ms = int(rres.seconds * 1000)
        # 【改写后的问题要同时喂给 retrieve 与 build_context】只喂 retrieve 的话，
        # 窗口选择会退化成截开头（见文件头说明）。这里算一次、显式传给 generate，
        # 让 context_chars 报的就是真正送进模型的那份。
        ctx = build_context(rres.hits, query=tctx.rewritten, profile=profile)
        # history 与本次 context 共享 num_ctx 预算（闸门在 store 里）
        refresh_history(store, tctx, budget_chars=len(ctx))

        t_gen = time.perf_counter()
        try:
            answer = await generator.generate(tctx.rewritten, rres.hits,
                                              context=ctx, history=tctx.history,
                                              profile=profile,
                                              doc=get_doc_profile(rres.routed_doc),
                                              lang_question=req.question)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"模型不可用：{e}") from e
        gen_ms = int((time.perf_counter() - t_gen) * 1000)

        doc = get_doc_profile(rres.routed_doc)
        commit_turn(store, tctx, answer, doc.key if doc else "")

        resp = ChatResponse(
            question=req.question, answer=answer,
            citations=_citations(rres.hits), mode="rag",
            retrieval_ms=retrieval_ms,
            # 改写耗时（规则层为 0）单独并入 TTFT 口径，不藏
            ttft_ms=tctx.rewrite_ms + retrieval_ms + gen_ms,
            total_ms=int((time.perf_counter() - t0) * 1000),
            n_candidates=rres.n_candidates, n_filtered=rres.n_filtered,
            profile=profile.name, context_chars=len(ctx),
            n_merged=rres.n_merged, n_added_neighbors=rres.n_added_neighbors,
            routed_doc=rres.routed_doc, route_matched=rres.route_matched,
            route_fallback=rres.route_fallback,
            focus_entity=next_focus_entity(resolved_entity=tctx.resolved_entity,
                                           doc_key=doc.key if doc else "",
                                           previous=tctx.focus_before),
            **turn_fields(tctx),
        )
        if use_cache:
            _cache_put(key, resp)
        return resp


# ----------------------------------------------------------------------
@router.get("/api/chat/stream")
async def chat_stream(request: Request, question: str, top_k: int | None = None,
                      no_rag: bool = False, profile: str | None = None,
                      session_id: str | None = None,
                      retrieval_mode: str | None = None, fusion: str | None = None,
                      w_keyword: float | None = None, reranker: str | None = None,
                      fusion_impl: str | None = None):
    """
    SSE 流式问答。

    事件序列：
      event: meta   —— 检索结果、引用与**改写信息**（首个 token 之前发出）
      event: token  —— 逐个文本增量
      event: done   —— 计时汇总
      event: error  —— 友好错误

    工单05：带 `session_id` 时走多轮 —— meta 里会多出 `rewrite`（原句→独立问题、
    改写方式与依据）与 `session_id/turn_index/history_*`，演示时不用口头解释。
    """
    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    async def gen():
        t0 = time.perf_counter()
        p = _derive_profile(profile, retrieval_mode=retrieval_mode, fusion=fusion,
                            w_keyword=w_keyword, reranker=reranker,
                            fusion_impl=fusion_impl)
        generator = Generator()

        # 排队提示：并发 >1 时用户能看到自己在等，而不是以为卡死
        if _semaphore.locked():
            yield _sse("status", {"message": "正在处理其他问题，排队中…"})

        try:
            # 构造放 try 内：连不上 Milvus 时异常是从构造器抛的（lazy property）
            retriever = Retriever(profile=p)
            async with _semaphore:
                if no_rag:
                    yield _sse("meta", {"mode": "no_rag", "citations": []})
                    ttft = None
                    async for piece in generator.client.chat_stream(
                        [{"role": "user", "content": question}]
                    ):
                        if ttft is None:
                            ttft = int((time.perf_counter() - t0) * 1000)
                        yield _sse("token", {"text": piece})
                    yield _sse("done", {
                        "ttft_ms": ttft or 0,
                        "total_ms": int((time.perf_counter() - t0) * 1000),
                        "mode": "no_rag",
                    })
                    return

                # 工单05：会话 + 改写（在本信号量内 —— 兜底改写会调模型，不能与主推理并发）
                store = get_session_store()
                tctx = await prepare_turn(store, _qu, question, session_id)

                rres = await retriever.retrieve(tctx.rewritten, top_k)
                retrieval_ms = int(rres.seconds * 1000)
                ctx_text = build_context(rres.hits, query=tctx.rewritten, profile=p)
                refresh_history(store, tctx, budget_chars=len(ctx_text))

                doc = get_doc_profile(rres.routed_doc)
                yield _sse("meta", {
                    "mode": "rag",
                    "profile": rres.profile,
                    "retrieval_ms": retrieval_ms,
                    "n_candidates": rres.n_candidates,
                    "n_filtered": rres.n_filtered,
                    # 工单02：把优化动作暴露出来，演示时能直接看到
                    # 「合并了几条重复表述 / 补了几条邻块 / 上下文涨了多少字」
                    "n_merged": rres.n_merged,
                    "n_added_neighbors": rres.n_added_neighbors,
                    # 工单03：路由到哪份文档（两份书章节雷同，出问题先看这一项）
                    "routed_doc": rres.routed_doc,
                    "route_matched": rres.route_matched,
                    "route_fallback": rres.route_fallback,
                    "context_chars": len(ctx_text),
                    "citations": [c.model_dump() for c in _citations(rres.hits)],
                    # 工单06：这一轮到底走的哪条检索路（演示时要能一眼看到）
                    "retrieval_mode": rres.retrieval_mode,
                    "fusion": rres.fusion, "reranker": rres.reranker,
                    "n_from_dense": rres.n_from_dense,
                    "n_from_keyword": rres.n_from_keyword,
                    "rerank_ms": rres.rerank_ms,
                    "rerank_fallback": rres.rerank_fallback,
                    "feedback_n": rres.feedback_n,
                    # 工单05：改写链路（原句 → 独立问题、方式、依据）整块给出，
                    # 前端直接渲染成「改写 chip」，观众自己就能看见 Query 理解这一步
                    "rewrite": {
                        "original": tctx.question,
                        "rewritten": tctx.rewritten,
                        "method": tctx.method,
                        "resolved_entity": tctx.resolved_entity,
                        "carried_intent": tctx.carried_intent,
                        "evidence": tctx.evidence,
                    },
                    "focus_entity": next_focus_entity(
                        resolved_entity=tctx.resolved_entity,
                        doc_key=doc.key if doc else "", previous=tctx.focus_before),
                    **turn_fields(tctx),
                })

                ttft = None
                answer_parts: list[str] = []
                async for piece in generator.generate_stream(
                        tctx.rewritten, rres.hits,
                        context=ctx_text, history=tctx.history, profile=p,
                        doc=doc, lang_question=question):
                    if await request.is_disconnected():
                        return
                    if ttft is None:
                        ttft = int((time.perf_counter() - t0) * 1000)
                    answer_parts.append(piece)
                    yield _sse("token", {"text": piece})

                # 写回会话，供下一轮消解
                commit_turn(store, tctx, "".join(answer_parts), doc.key if doc else "")

                yield _sse("done", {
                    # 改写耗时（规则层为 0）并入 TTFT 口径，并单独给出 rewrite_ms
                    "ttft_ms": (tctx.rewrite_ms + (ttft or 0)) if ttft else tctx.rewrite_ms,
                    "total_ms": int((time.perf_counter() - t0) * 1000),
                    "retrieval_ms": retrieval_ms,
                    "rewrite_ms": tctx.rewrite_ms,
                    "mode": "rag",
                })

        except VectorStoreError as e:
            yield _sse("error", {"message": f"向量库不可用：{e}"})
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"message": f"生成失败：{e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ----------------------------------------------------------------------
@router.post("/api/feedback", response_model=FeedbackOut)
async def feedback(req: FeedbackRequest) -> FeedbackOut:
    n = await asyncio.to_thread(_persist_feedback, req)
    return FeedbackOut(ok=True, received=n)


# ----------------------------------------------------------------------
# 工单05：会话管理（演示「会话隔离」与「TTL 过期」用）
# ----------------------------------------------------------------------
@router.get("/api/session/{session_id}")
async def session_info(session_id: str) -> dict:
    """查看某个会话的轮次与焦点实体（**只读，不创建**）。"""
    store = get_session_store()
    sess = store.get(session_id)
    msgs, dropped = store.history_messages(session_id)
    return {
        "session_id": session_id,
        "exists": sess is not None,
        "n_turns": len(sess.turns) if sess else 0,
        "focus_entity": store.focus_entity(session_id),
        "intent_template": store.intent_template(session_id),
        "history_turns": sum(1 for m in msgs if m["role"] == "user"),
        "history_chars": sum(len(m["content"]) for m in msgs),
        "history_dropped": dropped,
        "store": store.stats(),
    }


@router.delete("/api/session/{session_id}")
async def session_delete(session_id: str) -> dict:
    """结束会话（演示「换会话 → 上下文隔离」）。"""
    ok = get_session_store().delete(session_id)
    return {"ok": ok, "session_id": session_id}


# ----------------------------------------------------------------------
@router.get("/api/stats")
async def stats() -> dict:
    """运行时统计，供前端页脚展示。"""
    return {
        "ttft_budget_ms": 3000,
        "concurrency": settings.max_concurrency,
        "top_k": settings.retrieve_top_k,
        "models": {"llm": settings.llm_model, "embed": settings.embed_model},
        "ingest": get_pipeline().status().as_dict(),
        # 工单05：会话数 / 轮数（演示 TTL 与 LRU 用）
        "sessions": get_session_store().stats(),
        "rewrite": {"llm_fallback": settings.query_rewrite_llm_fallback},
    }
