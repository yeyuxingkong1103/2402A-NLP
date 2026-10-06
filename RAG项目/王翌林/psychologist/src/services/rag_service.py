"""RAG 在线问答编排：检索 → 提示词 → 大模型 → 后处理 → 记忆落库。

这是本目录最核心的编排层，一次问答会串起几乎所有下游服务：
  retrieval_service（改写+检索）→ memory_service（短期/长期记忆）
  → prompt_utils（拼提示词）→ llm_service（生成）→ crisis_service（危机兜底）
  → conversation_service（落库）→ memory_service（写回记忆）

被谁调用：api/rag 路由（非流式 answer、流式 answer_stream/SSE）。
依赖：core（配置/异常/日志）、db.redis（写锁）、rag.prompt/rag.chunker、上述各 service。

关键设计（逐条解释"为什么"）：
1) 分布式会话写锁 _conversation_write_lock：同一会话同一时刻只允许一条问答在途。
   否则用户连点两次发送时，两条问答会并发读改写 message_count 与短期记忆，造成消息乱序、
   计数被覆盖；锁用 Redis SET NX + TTL 实现，conversation_id 为空（自动新建会话）直接放行。
2) "先建会话、后检索"的两段式（prepare_basic / _retrieve_and_build）：
   SSE 首 token 优化——建会话+危机检测不依赖检索，可立刻下发 meta 事件，
   客户端先拿到 conversation_id 渲染骨架，再去等检索与生成，感知延迟明显下降。
3) 分阶段容错：
   - 流式生成失败：发 error 事件，但仍把已生成部分落库（不丢用户上下文）；
   - 生成内容为空：兜底一句"暂时无法生成回应"，避免前端出现空白气泡；
   - 落库失败：只记日志不中断 SSE（用户已看到答案，不该因写库失败再报错）；
   - 危机命中：回答末尾强制补齐转介提示（宁可重复，不可漏送）。
"""
import json
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.exceptions import RateLimitError
from src.core.logging import get_logger
from src.db import redis as redis_db
from src.rag import prompt as prompt_utils
from src.rag.chunker import estimate_tokens
from src.services import conversation_service, crisis_service, llm_service, memory_service
from src.services import persona_service, retrieval_service

logger = get_logger("rag")

_CONVERSATION_LOCK_TTL = 120  # 覆盖单次问答全流程（含 LLM 60s 超时）


@contextmanager
def _conversation_write_lock(conversation_id: Optional[int]):
    """会话写锁（需求 §9.4）：同一会话同一时刻只允许一条问答在途，防消息乱序/计数覆盖。

    conversation_id 为空表示自动新建会话（新 ID 无并发），直接放行。
    Redis 不可用时 acquire_lock 降级放行。
    """
    if not conversation_id:
        yield
        return
    key = redis_db.conversation_lock_key(conversation_id)
    if not redis_db.acquire_lock(key, ttl=_CONVERSATION_LOCK_TTL):
        raise RateLimitError("该会话正在处理上一条消息，请稍候再发")
    try:
        yield
    finally:
        # 无论正常结束还是异常/客户端断开，都要释放锁，否则该会话会被长期锁死
        redis_db.release_lock(key)


def _references(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把检索片段裁剪成"引用来源"精简结构下发给前端（文档、片段、来源、分数），
    不回传整段原文以免响应体过大；分数优先取重排分，没有重排则退回向量相似度分。"""
    return [
        {
            "doc_id": h.get("doc_id"),
            "chunk_id": h.get("chunk_id"),
            "source": h.get("source"),
            "score": round(float(h.get("rerank_score", h.get("score", 0.0))), 4),
        }
        for h in hits
    ]


def prepare_basic(db: Session, user_id: int, persona_id: int, message: str,
                  conversation_id: Optional[int] = None) -> Dict[str, Any]:
    """轻量上下文（阶段 4 SSE 首 token 优化）：建会话 + 历史 + 危机检测。

    这三步不依赖检索，SSE 在此之后即可发出 meta 事件，客户端立刻拿到
    conversation_id 并渲染会话骨架；检索与提示词拼装在 meta 之后进行。
    """
    start = time.time()
    conv = conversation_service.get_or_create_conversation(db, user_id, persona_id, conversation_id)
    persona = persona_service.get_persona_dict(db, persona_id)
    history = memory_service.get_short_term(db, user_id, persona_id, conv.id)
    crisis, crisis_keywords = crisis_service.detect_crisis(message)
    logger.info("会话与危机检测就绪 conversation=%s crisis=%s 耗时=%.0fms",
                conv.id, crisis, (time.time() - start) * 1000)
    return {
        "conversation": conv,
        "persona": persona,
        "history": history,
        "crisis": crisis,
        "crisis_keywords": crisis_keywords,
    }


def _retrieve_and_build(ctx: Dict[str, Any], user_id: int, persona_id: int,
                        message: str) -> Dict[str, Any]:
    """在 basic 上下文之上执行检索、长期记忆与提示词拼装，补全 ctx 并返回。"""
    start = time.time()
    # 知识库检索用改写后的 query（召回优先）；长期记忆检索用原问题（用户真实表述更贴近记忆摘要）
    hits, rewritten_query = retrieval_service.search_with_query_rewrite(
        persona_id, message, ctx["history"])
    memories = memory_service.get_long_term(user_id, persona_id, message)

    context = prompt_utils.build_context_block(hits)
    messages = prompt_utils.build_chat_prompt(
        persona=ctx["persona"], context=context, recent_messages=ctx["history"],
        question=message, long_term_memory=memories, crisis=ctx["crisis"],
    )
    ctx.update({
        "hits": hits,
        "memories": memories,
        "messages": messages,
        "rewritten_query": rewritten_query,
        "model_params": ctx["persona"].get("model_params") or {},
    })
    logger.info(
        "问答上下文就绪 user=%s persona=%s conversation=%s 命中=%d 改写=%s 耗时=%.0fms",
        user_id, persona_id, ctx["conversation"].id, len(hits),
        bool(rewritten_query != message), (time.time() - start) * 1000,
    )
    return ctx


def prepare(db: Session, user_id: int, persona_id: int, message: str,
            conversation_id: Optional[int] = None) -> Dict[str, Any]:
    """构造一次问答所需的全部上下文（检索 + 记忆 + 提示词）。非流式问答使用。"""
    ctx = prepare_basic(db, user_id, persona_id, message, conversation_id)
    return _retrieve_and_build(ctx, user_id, persona_id, message)


def _persist(db: Session, ctx: Dict[str, Any], user_id: int, message: str, answer: str,
             tokens: int = 0) -> None:
    """一次问答的收尾落库：MySQL 存问答两条消息（助手消息带引用），再写 Redis 短期记忆，
    最后按轮次阈值决定是否沉淀长期记忆。注意顺序——先落库拿到最新 message_count，
    maybe_save_long_term 才能正确判断是否触发（否则永远差一条）。"""
    conv = ctx["conversation"]
    persona = ctx["persona"]

    conversation_service.add_message(db, conv.id, "user", message, tokens=estimate_tokens(message))
    conversation_service.add_message(
        db, conv.id, "assistant", answer, tokens=tokens or estimate_tokens(answer),
        refs=_references(ctx["hits"]),
    )
    conversation_service.auto_title(db, conv, message)

    memory_service.append_short_term(user_id, persona["id"], conv.id, "user", message)
    memory_service.append_short_term(user_id, persona["id"], conv.id, "assistant", answer)
    db.refresh(conv)
    memory_service.maybe_save_long_term(db, conv, user_id, persona["id"])


def answer(db: Session, user_id: int, persona_id: int, message: str,
           conversation_id: Optional[int] = None) -> Dict[str, Any]:
    """非流式问答：加锁 → 准备上下文 → 调模型 → 危机后处理 → 落库，一次返回完整结果。
    锁从准备阶段一直持有到落库结束，保证同一会话不会被并发写坏。"""
    with _conversation_write_lock(conversation_id):
        ctx = prepare(db, user_id, persona_id, message, conversation_id)
        model_params = ctx["model_params"]
        result = llm_service.chat(
            ctx["messages"],
            temperature=model_params.get("temperature"),
            max_tokens=model_params.get("max_tokens"),
        )
        answer_text = crisis_service.post_process(result["content"], crisis=ctx["crisis"])
        _persist(db, ctx, user_id, message, answer_text, tokens=result["tokens"])
        return {
            "conversation_id": ctx["conversation"].id,
            "persona_id": persona_id,
            "answer": answer_text,
            "references": _references(ctx["hits"]),
            "tokens": result["tokens"],
            "finish_reason": result["finish_reason"],
            "crisis_detected": ctx["crisis"],
            "crisis_notice": crisis_service.crisis_notice() if ctx["crisis"] else None,
        }


def answer_stream(db: Session, user_id: int, persona_id: int, message: str,
                  conversation_id: Optional[int] = None) -> Generator[str, None, None]:
    """SSE 流式问答，事件类型：meta / delta / done / error。

    阶段 4 首 token 优化：meta 在建会话后立即发出（检索前），references 随
    done 事件下发；客户端首字节时间从「检索+改写完成」提前到「建会话完成」。
    """
    def event(event_type: str, payload: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    try:
        # 锁覆盖"建会话 → 检索 → 生成 → 落库"整个生命周期，直到 done 事件发出前才释放
        with _conversation_write_lock(conversation_id):
            basic = prepare_basic(db, user_id, persona_id, message, conversation_id)
            conv = basic["conversation"]
            # 先发 meta：客户端立刻拿到 conversation_id，无需等待检索与生成（首 token 优化）
            yield event("meta", {
                "conversation_id": conv.id,
                "persona_id": persona_id,
                "persona_name": basic["persona"].get("name"),
                "crisis_detected": basic["crisis"],
            })

            ctx = _retrieve_and_build(basic, user_id, persona_id, message)
            model_params = ctx["model_params"]
            collected: List[str] = []
            try:
                for piece in llm_service.chat_stream(
                    ctx["messages"],
                    temperature=model_params.get("temperature"),
                    max_tokens=model_params.get("max_tokens"),
                ):
                    collected.append(piece)
                    yield event("delta", {"content": piece})
            except Exception as exc:
                logger.error("流式生成失败：%s", exc, exc_info=True)
                yield event("error", {"message": f"大模型生成失败：{exc}"})

            raw_answer = "".join(collected)
            # 模型可能因超时/风控返回空串：兜底一句友好文案，保证 SSE 一定有正文
            if not raw_answer.strip():
                raw_answer = "抱歉，我这边暂时无法生成回应，请稍后再试。"
                yield event("delta", {"content": raw_answer})

            answer_text = crisis_service.post_process(raw_answer, crisis=ctx["crisis"])
            if ctx["crisis"] and crisis_service.crisis_notice().strip() not in answer_text:
                # 危机提示需完整送达前端
                tail = crisis_service.crisis_notice()
                yield event("delta", {"content": tail})
                answer_text = answer_text + tail

            tokens = estimate_tokens(answer_text)
            try:
                _persist(db, ctx, user_id, message, answer_text, tokens=tokens)
            except Exception as exc:
                # 用户已经流式看到完整回答，此处再抛错只会让前端认为失败，故仅记录日志
                logger.error("流式问答落库失败：%s", exc, exc_info=True)

            yield event("done", {
                "conversation_id": conv.id,
                "tokens": tokens,
                "finish_reason": "stop",
                "crisis_detected": ctx["crisis"],
                "references": _references(ctx["hits"]),
            })
    except RateLimitError as exc:
        # 会话并发锁被占用（S5）
        yield event("error", {"message": str(exc)})