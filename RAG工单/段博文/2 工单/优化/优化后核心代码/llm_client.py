# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""
大模型客户端（优化版）：DeepSeek API + 答案缓存 + 精简提示词。

核心优化点：
    1. 答案缓存：相同 query+上下文指纹 第二次直接返回缓存答案，跳过 API 调用；
    2. 精简系统提示词：去掉冗余要求，减少输入 token；
    3. max_tokens 2048→768：事实型问答不需要长篇大论，生成更快；
    4. temperature 0.3→0.1：回答更稳定，减少随机性带来的重复生成。
"""

import hashlib
import time
from functools import lru_cache
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    CACHE_ENABLED, CACHE_MAX_SIZE,
)
from logger import get_logger

logger = get_logger(__name__)

_llm: Optional[ChatOpenAI] = None

# 优化：精简系统提示词，减少输入 token，约束更聚焦
SYSTEM_PROMPT = (
    "你是基于知识库的PDF问答助手。严格依据【参考资料】回答用户问题；"
    "资料不足时说明「知识库中未找到相关内容」，不要编造。"
    "回答简洁，末尾标注出处（来源+页码）。"
)


def get_llm() -> ChatOpenAI:
    """获取 LLM 客户端单例。"""
    global _llm
    if _llm is None:
        if not LLM_API_KEY:
            raise RuntimeError("未配置 LLM_API_KEY（环境变量 deepseek_api_key1）")
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        logger.info(f"LLM 已加载：{LLM_MODEL} @ {LLM_BASE_URL} (max_tokens={LLM_MAX_TOKENS}, temp={LLM_TEMPERATURE})")
    return _llm


def _build_context(query: str, context_docs: List[Document]) -> str:
    """把检索到的文档列表拼成带编号的参考资料块 + 用户提问。"""
    if not context_docs:
        return "【参考资料】\n（无相关资料）\n\n【用户问题】\n" + query

    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page_number", 0)
        page_tag = f", 第{page}页" if page else ""
        blocks.append(f"[{i}](来源:{source}{page_tag}) {doc.page_content}")
    context_text = "\n\n".join(blocks)

    return (
        f"【参考资料】\n{context_text}\n\n"
        f"【用户问题】\n{query}"
    )


def _context_fingerprint(query: str, context_docs: List[Document]) -> str:
    """生成上下文指纹用于缓存键（query + 各文档 page_content 前 50 字）。"""
    parts = [query] + [d.page_content[:50] for d in context_docs]
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


# ==================== 新增：答案缓存 ====================
@lru_cache(maxsize=CACHE_MAX_SIZE if CACHE_ENABLED else 0)
def _generate_cached(fingerprint: str, query: str, context_text: str) -> str:
    """带缓存的内部生成函数（实际调用 LLM）。"""
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context_text),
    ]
    t0 = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - t0
    logger.info(f"[LLM计时] 生成回答：query='{query[:30]}'，长度={len(response.content)}，耗时={elapsed:.2f}s")
    return response.content


def generate_answer(query: str, context_docs: List[Document]) -> str:
    """
    基于检索上下文生成回答（带缓存）。

    优化：开启 CACHE_ENABLED 时，相同 query+上下文指纹 直接命中缓存返回。
    """
    user_message = _build_context(query, context_docs)

    if CACHE_ENABLED:
        fp = _context_fingerprint(query, context_docs)
        t0 = time.time()
        answer = _generate_cached(fp, query, user_message)
        if time.time() - t0 < 0.1:
            logger.info(f"[缓存命中] 答案查询 '{query[:30]}'，耗时 {time.time()-t0:.3f}s")
        return answer
    else:
        llm = get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        t0 = time.time()
        response = llm.invoke(messages)
        logger.info(f"[LLM计时] 生成回答：query='{query[:30]}'，长度={len(response.content)}，耗时={time.time()-t0:.2f}s")
        return response.content


def stream_answer(query: str, context_docs: List[Document]):
    """流式生成回答：逐 chunk 返回，前端用 SSE 接收。"""
    llm = get_llm()
    user_message = _build_context(query, context_docs)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
