# -*- coding: utf-8 -*-
"""RAG 主链路编排：

    查询改写 → 混合检索(dense∥sparse→RRF) → 精排 → 人格注入 → 流式生成 → 后处理

设计文档 §2.2 的实现。
"""
import re
import time
from collections.abc import Iterator

from ..core import config
from ..core.logging import get_logger
from ..models import Character
from . import llm, long_memory, persona, rerank, retrieval

log = get_logger("rag_chain")


# ---------------------------------------------------------------- 检索阶段
def prepare(character: Character, question: str, memory: list[dict]) -> dict:
    """执行检索侧的全部工作：改写 → 召回 → 精排。"""
    t_start = time.time()
    trace: dict = {}

    # ---- ① 查询改写（指代消解） ----
    t = time.time()
    rewritten = question
    if config.REWRITE_ENABLED and memory:
        rewritten = llm.rewrite_query(question, memory)
    trace["rewritten_query"] = rewritten
    trace["rewrite_ms"] = round((time.time() - t) * 1000)
    trace["rewrite_skipped"] = (rewritten == question and bool(memory))

    # ---- ② 混合检索 ----
    t = time.time()
    hits, _ = retrieval.search(
        rewritten,
        collection=character.kb_collection,
        recall_k=character.recall_top_k,
    )
    trace["recall"] = len(hits)
    trace["recall_ms"] = round((time.time() - t) * 1000)

    # ---- ③ 精排 ----
    t = time.time()
    if hits and config.RERANK_ENABLED:
        hits = rerank.rerank(rewritten, hits, character.rerank_top_k)
    else:
        hits = hits[:character.rerank_top_k]
    trace["reranked"] = len(hits)
    trace["rerank_ms"] = round((time.time() - t) * 1000)

    trace["total_retrieval_ms"] = round((time.time() - t_start) * 1000)
    log.info("检索完成 | 召回 %d -> 精排 %d | 改写%.0fms 召回%.0fms 精排%.0fms",
             trace["recall"], trace["reranked"],
             trace["rewrite_ms"], trace["recall_ms"], trace["rerank_ms"])

    return {"rewritten": rewritten, "hits": hits, "trace": trace}


def _safe_recall(character, question: str) -> list[dict]:
    """召回跨会话长期记忆。失败返回空 —— 增强能力不可用不应影响主对话。

    ⚠️ **排除当前会话**（`exclude_conversation`）：当前会话的历史已由短期记忆
    （Redis，最近 6 轮）提供，长期记忆再召回一遍是重复注入，会白占上下文、
    并挤掉真正有用的**跨会话**信息。
    初版漏传了该参数，实测召回结果里混进了当前会话自己的上一轮。
    """
    try:
        uid = getattr(character, "_user_id", None)
        if uid is None:
            return []
        return long_memory.recall(uid, question,
                                  exclude_conversation=getattr(character, "_conversation_id", None))
    except Exception as e:
        log.warning("长期记忆召回异常（已忽略）: %s", str(e)[:150])
        return []


# ---------------------------------------------------------------- 后处理
_DECORATION_RE = re.compile(r"^(?:。|，|、|\s)+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def postprocess(text: str, character: Character) -> str:
    """后处理：清理首尾装饰、压缩空行、规范引用编号。"""
    if not text:
        return text
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _DECORATION_RE.sub("", text)
    text = re.sub(r"\[(\d+)\](?:\s*\[(\d+)\])+",
                  lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(0))),
                  text)
    return text.strip()


# ---------------------------------------------------------------- 非流式
def ask(character: Character, question: str, memory: list[dict]) -> dict:
    """完整问答（非流式）。"""
    t0 = time.time()
    prepared = prepare(character, question, memory)
    messages = persona.build_messages(character, question, prepared["hits"], memory,
                                      _safe_recall(character, question))

    t = time.time()
    answer = llm.chat(messages, temperature=character.temperature)
    gen_ms = round((time.time() - t) * 1000)

    answer = postprocess(answer, character)
    prepared["trace"]["generate_ms"] = gen_ms
    prepared["trace"]["total_ms"] = round((time.time() - t0) * 1000)

    return {
        "answer": answer,
        "sources": persona.build_sources(prepared["hits"]),
        "trace": prepared["trace"],
    }


# ---------------------------------------------------------------- 流式
def ask_stream(character: Character, question: str,
               memory: list[dict]) -> Iterator[tuple[str, dict]]:
    """流式问答，逐事件 yield (event_name, data)。

    事件顺序：trace → sources → delta* → done
    """
    t0 = time.time()
    try:
        prepared = prepare(character, question, memory)
    except Exception as e:
        log.exception("检索阶段失败")
        yield "error", {"code": "RETRIEVAL_FAILED", "message": str(e)[:200]}
        return

    yield "trace", prepared["trace"]
    yield "sources", persona.build_sources(prepared["hits"])

    messages = persona.build_messages(character, question, prepared["hits"], memory,
                                      _safe_recall(character, question))
    t = time.time()
    buf: list[str] = []
    try:
        for delta in llm.chat_stream(messages, temperature=character.temperature):
            buf.append(delta)
            yield "delta", {"text": delta}
    except Exception as e:
        log.exception("生成阶段失败")
        yield "error", {"code": "LLM_FAILED", "message": str(e)[:200]}
        return

    gen_ms = round((time.time() - t) * 1000)
    answer = postprocess("".join(buf), character)
    yield "done", {
        "answer": answer,
        "generate_ms": gen_ms,
        "total_ms": round((time.time() - t0) * 1000),
    }
