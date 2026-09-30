# -*- coding: utf-8 -*-
"""LangChain（LCEL）版 RAG 链路 —— 与手写 rag_chain 并存，由配置切换。

对照手写链路的差异：
    检索段：RunnablePassthrough.assign + HybridRetriever（LangChain 接口）
    生成段：ChatPromptTemplate | ChatOllama | StrOutputParser

保持一致的部分（刻意复用，不重写）：
    精排    —— LangChain 无 bge-reranker 集成，仍用 rerank.rerank
    后处理  —— 仍用 rag_chain.postprocess，保证两条链路输出格式一致
    Prompt  —— 仍用 persona.render_system，避免模板漂移

事件协议与 rag_chain.ask_stream 完全相同（trace → sources → delta* → done）。
"""
import time
from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from ..core import config
from ..core.logging import get_logger
from ..models import Character
from . import long_memory, persona, rerank
from .lc_retrievers import HybridRetriever, doc_to_hit
from .rag_chain import postprocess

log = get_logger("lc_chain")


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


def _ollama_base_url() -> str:
    """config.OLLAMA_BASE_URL 是 OpenAI 兼容端点(带 /v1)，ChatOllama 需要裸地址。"""
    return config.OLLAMA_BASE_URL.removesuffix("/v1")


def _build_llm(character: Character, streaming: bool = False) -> ChatOllama:
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=_ollama_base_url(),
        temperature=character.temperature if character.temperature is not None
        else config.LLM_TEMPERATURE,
        num_predict=config.LLM_MAX_TOKENS,
        streaming=streaming,
    )


def _prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human", "{question}"),
    ])


def _rerank_step(character: Character):
    """把精排插进 LCEL 管道 —— 框架未提供该集成，故用 RunnableLambda。"""

    def _fn(payload: dict) -> dict:
        hits = [doc_to_hit(d) for d in payload["hits"]]
        recalled = len(hits)
        t = time.time()
        if hits and config.RERANK_ENABLED:
            hits = rerank.rerank(payload["question"], hits, character.rerank_top_k)
        else:
            hits = hits[:character.rerank_top_k]
        # LCEL 把召回与精排融合在一个管道里，需显式带出两个数字，
        # 否则 trace 里的召回数/耗时会被精排污染，前端链路可视就失真了。
        return {**payload, "hits": hits, "recall_n": recalled,
                "rerank_ms": round((time.time() - t) * 1000)}

    return RunnableLambda(_fn)


def _retrieval_chain(character: Character):
    """LCEL 检索段：混合检索 → 精排。"""
    retriever = HybridRetriever(
        collection=character.kb_collection,
        k=character.recall_top_k,
    )
    return (
        RunnablePassthrough.assign(
            hits=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | _rerank_step(character)
    )


# ---------------------------------------------------------------- 检索阶段
def prepare(character: Character, question: str, memory: list[dict]) -> dict:
    """执行检索侧全部工作：改写 → 召回 → 精排（与 rag_chain.prepare 同签名）。"""
    from . import llm

    t_start = time.time()
    trace: dict = {}

    t = time.time()
    rewritten = question
    if config.REWRITE_ENABLED and memory:
        rewritten = llm.rewrite_query(question, memory)
    trace["rewritten_query"] = rewritten
    trace["rewrite_ms"] = round((time.time() - t) * 1000)
    trace["rewrite_skipped"] = (rewritten == question and bool(memory))

    t = time.time()
    chain = _retrieval_chain(character)
    out = chain.invoke({"question": rewritten, "memory": memory})
    hits = out["hits"]
    rerank_ms = out.get("rerank_ms", 0)
    total_ms = round((time.time() - t) * 1000)
    trace["recall"] = out.get("recall_n", len(hits))   # 精排前的召回数
    trace["recall_ms"] = max(0, total_ms - rerank_ms)  # 抵扣精排耗时，与手动链路口径一致
    trace["reranked"] = len(hits)
    trace["rerank_ms"] = rerank_ms
    trace["total_retrieval_ms"] = round((time.time() - t_start) * 1000)
    trace["chain"] = "langchain"

    log.info("LCEL 检索完成 | 精排 %d 条 | %.0fms",
             len(hits), trace["total_retrieval_ms"])
    return {"rewritten": rewritten, "hits": hits, "trace": trace}


# ---------------------------------------------------------------- 非流式
def ask(character: Character, question: str, memory: list[dict]) -> dict:
    """完整问答（非流式）。"""
    t0 = time.time()
    prepared = prepare(character, question, memory)

    t = time.time()
    chain = (
        _prompt()
        | _build_llm(character)
        | StrOutputParser()
    )
    long_mem = _safe_recall(character, question)
    answer = chain.invoke({
        "system": persona.render_system(character, question, prepared["hits"],
                                        memory, long_mem),
        "question": question,
    })
    prepared["trace"]["generate_ms"] = round((time.time() - t) * 1000)
    prepared["trace"]["total_ms"] = round((time.time() - t0) * 1000)

    return {
        "answer": postprocess(answer, character),
        "sources": persona.build_sources(prepared["hits"]),
        "trace": prepared["trace"],
    }


# ---------------------------------------------------------------- 流式
def ask_stream(character: Character, question: str,
               memory: list[dict]) -> Iterator[tuple[str, dict]]:
    """流式问答，事件协议与 rag_chain.ask_stream 完全一致。"""
    t0 = time.time()
    try:
        prepared = prepare(character, question, memory)
    except Exception as e:
        log.exception("LCEL 检索阶段失败")
        yield "error", {"code": "RETRIEVAL_FAILED", "message": str(e)[:200]}
        return

    yield "trace", prepared["trace"]
    yield "sources", persona.build_sources(prepared["hits"])

    system = persona.render_system(character, question, prepared["hits"],
                                   memory, _safe_recall(character, question))
    chain = _prompt() | _build_llm(character, streaming=True) | StrOutputParser()

    t = time.time()
    buf: list[str] = []
    try:
        for chunk in chain.stream({"system": system, "question": question}):
            buf.append(chunk)
            yield "delta", {"text": chunk}
    except Exception as e:
        log.exception("LCEL 生成阶段失败")
        yield "error", {"code": "LLM_FAILED", "message": str(e)[:200]}
        return

    gen_ms = round((time.time() - t) * 1000)
    prepared["trace"]["generate_ms"] = gen_ms
    prepared["trace"]["total_ms"] = round((time.time() - t0) * 1000)
    yield "done", {
        "answer": postprocess("".join(buf), character),
        "generate_ms": gen_ms,
        "total_ms": prepared["trace"]["total_ms"],
    }
