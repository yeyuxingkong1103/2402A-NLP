"""
chat.py — 核心对话调度

按领域把请求分派给 domains 下的对应类，串起整条链路：

    短期记忆 → 领域路由 → Query 改写 → 并行检索（知识库 + 长期记忆）
      → 四段式提示词 → 流式生成 → 存短期记忆 → 存长期记忆 → 提取关系记忆

对外提供两个入口：
    chat()         一次性返回完整结果
    chat_stream()  逐段产出事件，供 FastAPI SSE 与 Streamlit 共用

chat_stream 产出的事件类型：
    {"type": "meta"}   领域、角色、命中的资料来源（先于正文发出）
    {"type": "delta"}  正文增量
    {"type": "done"}   结束，带完整回复与耗时
    {"type": "error"}  失败原因
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterator

import config
import domains
import intent
import long_term
import session_memory

_logger = None


def _log():
    global _logger
    if _logger is None:
        from loguru import logger

        _logger = logger
    return _logger


def _safe(future: Future, what: str) -> list[dict]:
    """取检索结果。任一条检索失败都不该让整轮对话挂掉。"""
    try:
        return future.result() or []
    except Exception as exc:
        _log().warning(f"{what}失败，本轮不注入这部分内容：{exc}")
        return []


def _parallel_retrieve(dom, query: str, user_id: str) -> tuple[list[dict], list[dict]]:
    """知识库检索与长期记忆检索互不依赖，并行跑省一次往返。

    两条路径都会用到向量模型，embeddings 里的懒加载已加锁，
    不会各自加载一份 2.3GB 权重。
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        kb_future = pool.submit(dom.retrieve, query, config.KB_TOP_K)
        mem_future = pool.submit(
            long_term.search_long_term, user_id, query, dom.domain, config.LONG_TERM_TOP_K
        )
        return _safe(kb_future, "知识库检索"), _safe(mem_future, "长期记忆检索")


def prepare(user_id: str, user_input: str) -> dict:
    """对话前的准备工作：记忆、领域路由、改写、检索、提示词组装。"""
    started = time.time()
    history = session_memory.get_short_term(user_id)

    # 领域路由：决定这一轮交给法律 / 医疗 / 英语 / 闲聊中的哪一块
    dom = domains.route(user_input, history)

    docs: list[dict] = []
    memory_hits: list[dict] = []
    rewritten = user_input
    if dom.domain != "chat":
        rewritten = intent.rewrite_query(user_input, history)
        docs, memory_hits = _parallel_retrieve(dom, rewritten, user_id)

    # 长期记忆必须进提示词，召回为空时由模板渲染成"暂无历史记忆"
    memory_text = long_term.format_hits(memory_hits)
    system_prompt = dom.build_system_prompt(docs, memory_text, history)

    _log().info(
        f"用户请求 user={user_id} domain={dom.domain} role={dom.role_name} "
        f"docs={len(docs)} memory={len(memory_hits)} query={user_input[:60]}"
    )

    return {
        "user_id": user_id,
        "user_input": user_input,
        "history": history,
        "domain": dom.domain,
        "role": dom.role_name,
        "domain_obj": dom,
        "rewritten_query": rewritten,
        "docs": docs,
        "memory_hits": memory_hits,
        "sources": dom.sources(docs),
        # 一条资料都没检索到时标 False，前端据此提示"本轮回答没有资料支撑"
        "grounded": bool(docs) or dom.domain == "chat",
        "system_prompt": system_prompt,
        "prepare_seconds": round(time.time() - started, 3),
    }


def _persist(ctx: dict, reply: str) -> None:
    """写入短期记忆，并把本轮摘要沉淀到长期记忆。"""
    user_id = ctx["user_id"]
    user_input = ctx["user_input"]

    try:
        session_memory.save_short_term(user_id, user_input, reply)
    except Exception as exc:
        _log().warning(f"写入短期记忆失败：{exc}")

    if not reply.strip():
        return

    try:
        text = f"用户问题：{user_input}\n回答要点：{reply[:200]}"
        long_term.store_long_term(
            user_id,
            text,
            role=ctx["role"],
            fact_type="summary",
            source=ctx["domain"],
            domain=ctx["domain"],
        )
    except Exception as exc:
        _log().warning(f"写入长期记忆失败：{exc}")

    # 每 PROFILE_EXTRACT_EVERY 轮提炼一次偏好与已确认事实。这是额外一次模型往返
    # （数秒），放后台线程跑，否则每第 5 轮会明显变慢。
    try:
        threading.Thread(
            target=long_term.maybe_extract_profile,
            args=(user_id, ctx["domain"]),
            daemon=True,
        ).start()
    except Exception as exc:
        _log().warning(f"启动关系记忆提取失败：{exc}")


# 模型偶尔会把系统提示词里的自我描述抄进正文，连同后面的标点一起删掉。
# 身份词用 + 而非 * —— 至少要命中一个，否则"作为""我是"单独出现时
# 会误删正常中文（"作为劳动者，你有权……"）。
_CLICHE = re.compile(
    r"(作为|我是)(一个|一名|一位)?"
    r"(?:AI|人工智能|大?语言模型|智能助手|智能助理|助手|助理)+[，,。.！!：:]?[ \t]*"
)

# 中文语境里的半角标点转全角。只在标点左边是汉字时才转，
# 否则会把英语领域的例句 "look forward to your reply," 一起改坏。
_CJK = r"[一-鿿]"
_PUNCT_MAP = {",": "，", ";": "；", "?": "？", "!": "！", ":": "："}


def _post_process(reply: str) -> str:
    """清理模型输出的排版噪声，返回可直接展示的正文。

    处理四类问题：自我描述套话、中文里的半角标点、连续空格、多余空行。
    只做展示层的修剪，不增删事实内容。
    """
    if not reply:
        return reply

    text = _CLICHE.sub("", reply)
    for half, full in _PUNCT_MAP.items():
        text = re.sub(rf"(?<={_CJK}){re.escape(half)}", full, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 逐行去掉行尾空格，行首不动，保留模型有意写的缩进
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def chat(user_id: str, user_input: str, save_memory: bool = True) -> dict:
    """非流式对话，一次性返回完整回复。"""
    ctx = prepare(user_id, user_input)
    started = time.time()

    reply = _post_process(ctx["domain_obj"].answer(ctx["system_prompt"], user_input, ctx["history"]))

    if save_memory and reply:
        _persist(ctx, reply)

    return {
        "reply": reply,
        "domain": ctx["domain"],
        "role": ctx["role"],
        "rewritten_query": ctx["rewritten_query"],
        "sources": ctx["sources"],
        "grounded": ctx["grounded"],
        "elapsed": round(time.time() - started, 3),
    }


def chat_stream(user_id: str, user_input: str, save_memory: bool = True) -> Iterator[dict]:
    """流式对话，逐段产出事件字典。"""
    try:
        ctx = prepare(user_id, user_input)
    except Exception as exc:
        _log().error(f"对话准备阶段失败：{exc}")
        yield {"type": "error", "message": f"对话初始化失败：{exc}"}
        return

    # 先把领域和来源发给前端，用户能立刻看到"正在查什么"
    yield {
        "type": "meta",
        "domain": ctx["domain"],
        "role": ctx["role"],
        "rewritten_query": ctx["rewritten_query"],
        "sources": ctx["sources"],
        "grounded": ctx["grounded"],
        "prepare_seconds": ctx["prepare_seconds"],
    }

    started = time.time()
    parts: list[str] = []
    try:
        for piece in ctx["domain_obj"].generate(
            ctx["system_prompt"], user_input, ctx["history"]
        ):
            parts.append(piece)
            yield {"type": "delta", "content": piece}
    except Exception as exc:
        _log().error(f"模型生成失败：{exc}")
        yield {"type": "error", "message": f"模型生成失败：{exc}"}
        return

    reply = _post_process("".join(parts))
    if not reply:
        # 推理模型把 max_tokens 全用在思维链上时正文会为空，必须明说，不能让前端显示一片空白
        _log().warning("模型返回空回答，通常是思维链占满了 max_tokens")
        yield {"type": "error", "message": "模型没有返回内容，请重试；若持续出现请调大 LLM_MAX_TOKENS"}
        return

    if save_memory:
        _persist(ctx, reply)

    yield {
        "type": "done",
        "reply": reply,
        "domain": ctx["domain"],
        "role": ctx["role"],
        "sources": ctx["sources"],
        "grounded": ctx["grounded"],
        "elapsed": round(time.time() - started, 3),
    }
