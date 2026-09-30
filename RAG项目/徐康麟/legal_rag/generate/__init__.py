# -*- coding: utf-8 -*-
"""生成层：提示词、大模型客户端（mock / ollama / deepseek / openai 兼容）、路由、后处理。"""
from .llm_base import LLMClient, LLMError, build_llm_client
from .llm_metrics import LLMStats, record_llm_stats
from .ollama import OllamaClient, OllamaError, resolve_ollama_model
from .postprocess import postprocess
from .prompt import build_context_block, build_messages, build_system_prompt
from .router import ModelRouter

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMStats",
    "OllamaClient",
    "OllamaError",
    "build_llm_client",
    "ModelRouter",
    "record_llm_stats",
    "resolve_ollama_model",
    "build_system_prompt",
    "build_context_block",
    "build_messages",
    "postprocess",
]
