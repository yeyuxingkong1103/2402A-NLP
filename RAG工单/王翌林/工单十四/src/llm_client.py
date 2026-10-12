# -*- coding: utf-8 -*-
"""
工单十四：DeepSeek LLM 客户端
"""
import os
import time
from typing import Dict, List
from loguru import logger

_client = None

def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key.startswith("your_key"):
            raise RuntimeError("请在 .env 中设置 DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        _client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
        logger.info(f"DeepSeek client 就绪: {base_url}")
    return _client

def chat(messages, model=None, temperature=0.3, max_tokens=2048, disable_thinking=True, **kwargs):
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    client = get_client()
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}} if disable_thinking else None,
            **kwargs)
    except Exception as e:
        if disable_thinking and ("thinking" in str(e) or "400" in str(e) or "extra_body" in str(e)):
            logger.warning(f"thinking 参数不受支持，回退普通调用: {str(e)[:80]}")
            resp = client.chat.completions.create(model=model, messages=messages,
                                                  temperature=temperature, max_tokens=max_tokens, **kwargs)
        else:
            raise
    latency_ms = (time.time() - t0) * 1000
    usage = resp.usage
    result = {"content": resp.choices[0].message.content or "", "latency_ms": round(latency_ms, 1),
              "token_usage": {"prompt_tokens": usage.prompt_tokens if usage else 0,
                              "completion_tokens": usage.completion_tokens if usage else 0,
                              "total_tokens": usage.total_tokens if usage else 0},
              "model": model}
    logger.info(f"LLM 返回: model={model}, latency={result['latency_ms']:.0f}ms, tokens={result['token_usage']['total_tokens']}")
    return result
