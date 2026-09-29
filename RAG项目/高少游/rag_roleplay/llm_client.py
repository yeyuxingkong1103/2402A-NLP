# -*- coding: utf-8 -*-
"""
大模型客户端：支持在线 API（DeepSeek/千问/Claude/GPT/Gemini/豆包/硅基流动）
和本地部署（vLLM/SGLang/xInference），统一走 OpenAI 兼容接口
"""

import os  # 环境变量
from typing import Optional  # 类型标注

from langchain_openai import ChatOpenAI  # 所有 OpenAI 兼容接口都走这个类
from langchain_core.messages import SystemMessage, HumanMessage  # 对话消息

from config import (  # 配置
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL,
)
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# LLM 单例：按 (base_url, model) 缓存，切换模型时不复用旧实例
_llm_instances: dict = {}


def get_llm(temperature: float = None, use_local: bool = False) -> ChatOpenAI:
    """
    获取 LLM 客户端单例

    Args:
        temperature: 温度（不传用配置默认值；角色专属温度可通过 role 表覆盖）
        use_local: True=用本地部署模型；False=用在线 API（默认）
    """
    if use_local or LOCAL_LLM_BASE_URL:  # 用本地部署
        key = ("local", LOCAL_LLM_MODEL, temperature or LLM_TEMPERATURE)
        if key not in _llm_instances:
            _llm_instances[key] = ChatOpenAI(
                model=LOCAL_LLM_MODEL,
                api_key="not-needed",  # 本地部署不需要 key，占位即可
                base_url=LOCAL_LLM_BASE_URL,
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            logger.info(f"本地 LLM 已加载：{LOCAL_LLM_MODEL} @ {LOCAL_LLM_BASE_URL}")
        return _llm_instances[key]
    else:  # 在线 API
        key = ("online", LLM_MODEL, temperature or LLM_TEMPERATURE)
        if key not in _llm_instances:
            if not LLM_API_KEY:
                raise RuntimeError("未配置 LLM_API_KEY（环境变量 deepseek_api_key1）")
            _llm_instances[key] = ChatOpenAI(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            logger.info(f"在线 LLM 已加载：{LLM_MODEL} @ {LLM_BASE_URL}")
        return _llm_instances[key]


def chat(system_prompt: str, history: list, user_message: str,
         temperature: float = None, use_local: bool = False) -> str:
    """
    多轮对话：系统提示词 + 历史消息 + 当前用户消息 → LLM 回答

    Args:
        system_prompt: 角色系统提示词（定义人格、工作原则）
        history:      历史消息列表 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        user_message: 当前用户消息（含检索上下文）
        temperature:  温度
        use_local:    是否用本地部署模型
    Returns:
        LLM 回答文本
    """
    llm = get_llm(temperature=temperature, use_local=use_local)  # 获取 LLM
    messages = [SystemMessage(content=system_prompt)]  # 系统提示词（第一条）
    # 把历史消息加上
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))  # 用户消息
        elif msg["role"] == "assistant":
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))  # 角色回答
    messages.append(HumanMessage(content=user_message))  # 当前消息
    response = llm.invoke(messages)  # 调用 LLM（同步阻塞）
    return response.content  # 返回回答文本


def stream_chat(system_prompt: str, history: list, user_message: str,
                temperature: float = None, use_local: bool = False):
    """
    流式对话：逐 token 返回，前端用 SSE 接收

    Yields:
        每个 chunk 的文本内容
    """
    llm = get_llm(temperature=temperature, use_local=use_local)
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))
    for chunk in llm.stream(messages):  # 流式生成
        if chunk.content:
            yield chunk.content  # 逐 chunk 返回
