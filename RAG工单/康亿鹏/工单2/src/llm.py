# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：大语言模型模块。封装 DeepSeek（OpenAI 兼容接口），
          为答案生成与 Query 理解提供统一的 LLM 实例。
"""
from langchain_openai import ChatOpenAI

import config
from src.utils import logger


class LLMConfigError(Exception):
    """LLM 配置异常（缺少 API Key 等）。"""


def check_llm_config():
    """校验 LLM 配置，缺失时给出明确的修复指引（容错机制）。"""
    if not config.LLM_API_KEY or config.LLM_API_KEY.startswith("sk-xxxx"):
        raise LLMConfigError(
            "未配置 DeepSeek API Key。请复制 .env.example 为 .env，并填写 DEEPSEEK_API_KEY1。"
        )


def get_llm(streaming: bool = False, temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    """获取 DeepSeek 对话模型实例。"""
    check_llm_config()
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        timeout=config.LLM_TIMEOUT,
        streaming=streaming,
    )
    return llm


_llm_cache = {}


def get_cached_llm(streaming: bool = False) -> ChatOpenAI:
    """按 streaming 维度缓存 LLM 实例。"""
    key = f"llm_{streaming}"
    if key not in _llm_cache:
        _llm_cache[key] = get_llm(streaming=streaming)
        logger.info("DeepSeek LLM 初始化完成：model=%s", config.LLM_MODEL)
    return _llm_cache[key]
