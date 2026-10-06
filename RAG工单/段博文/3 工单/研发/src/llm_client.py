# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
LLM 客户端模块：DeepSeek API 调用，支持流式和非流式。

优化点：
    1. 答案缓存：LLM 生成结果缓存，重复问题直接返回；
    2. 流式支持：支持 SSE 流式输出，提升用户体验；
    3. 表格理解：Prompt 中强调表格结构理解能力。
"""

import json
import time
from collections import OrderedDict
from typing import Generator, List, Optional

import httpx
from langchain_core.documents import Document

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    CACHE_ENABLED, CACHE_MAX_SIZE,
)
from logger import get_logger

logger = get_logger(__name__)


class LLMCache:
    """LLM 答案 LRU 缓存。"""

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _get_key(self, query: str, context: str) -> str:
        """生成缓存键。"""
        content = f"{query}_{context[:200]}"
        return content

    def get(self, query: str, context: str) -> Optional[str]:
        """获取缓存答案。"""
        key = self._get_key(query, context)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            logger.debug(f"LLM 缓存命中：{query[:30]}...")
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, query: str, context: str, answer: str):
        """设置缓存答案。"""
        key = self._get_key(query, context)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = answer
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        """清空缓存。"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_info(self) -> dict:
        """获取缓存统计。"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / (self.hits + self.misses) * 100, 2) if (self.hits + self.misses) > 0 else 0,
        }


# 全局缓存实例
_llm_cache = LLMCache(CACHE_MAX_SIZE)


def _build_prompt(query: str, documents: List[Document]) -> str:
    """构建 Prompt。"""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        block_type = doc.metadata.get("block_type", "text")
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page_number", 0)

        # 表格块特殊标记
        if block_type == "table":
            context_parts.append(f"[表格 {i}（来源：{source} 第{page}页）]\n{doc.page_content}")
        else:
            context_parts.append(f"[资料 {i}（来源：{source} 第{page}页）]\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    prompt = f"""基于以下参考资料回答问题。请仔细分析资料内容，特别是表格数据，给出准确、完整的答案。

{context}

问题：{query}

要求：
1. 仔细分析参考资料，特别是表格中的数据
2. 如果参考资料包含表格，请正确理解表格结构（行列关系）
3. 答案要准确、完整，包含所有关键信息
4. 如果资料不足以回答问题，请说明"根据提供的资料无法回答"

答案："""

    return prompt


def generate_answer(query: str, documents: List[Document]) -> str:
    """生成答案（非流式）。

    Args:
        query: 用户问题
        documents: 检索到的文档列表

    Returns:
        str: 生成的答案
    """
    if not documents:
        return "抱歉，没有找到相关的资料来回答您的问题。"

    # 构建上下文用于缓存
    context = "\n".join([d.page_content[:200] for d in documents])

    # 检查缓存
    if CACHE_ENABLED:
        cached_answer = _llm_cache.get(query, context)
        if cached_answer is not None:
            logger.info(f"LLM 缓存命中：{query[:30]}...")
            return cached_answer

    # 构建 Prompt
    prompt = _build_prompt(query, documents)

    # 调用 API
    start_time = time.time()
    try:
        response = _call_deepseek_api(prompt)
        elapsed = time.time() - start_time
        logger.info(f"LLM 生成完成：耗时 {elapsed:.2f}s，答案长度 {len(response)}")

        # 缓存结果
        if CACHE_ENABLED and response:
            _llm_cache.set(query, context, response)

        return response
    except Exception as e:
        logger.error(f"LLM 生成失败：{e}")
        raise


def stream_answer(query: str, documents: List[Document]) -> Generator[str, None, None]:
    """生成答案（流式）。

    Args:
        query: 用户问题
        documents: 检索到的文档列表

    Yields:
        str: 答案片段
    """
    if not documents:
        yield "抱歉，没有找到相关的资料来回答您的问题。"
        return

    # 构建 Prompt
    prompt = _build_prompt(query, documents)

    # 调用流式 API
    start_time = time.time()
    try:
        for chunk in _call_deepseek_api_stream(prompt):
            yield chunk
        elapsed = time.time() - start_time
        logger.info(f"LLM 流式生成完成：耗时 {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"LLM 流式生成失败：{e}")
        yield f"\n\n[生成失败：{e}]"


def _call_deepseek_api(prompt: str) -> str:
    """调用 DeepSeek API（非流式）。"""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个专业的文档问答助手，擅长分析表格数据。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def _call_deepseek_api_stream(prompt: str) -> Generator[str, None, None]:
    """调用 DeepSeek API（流式）。"""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个专业的文档问答助手，擅长分析表格数据。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": True,
    }

    with httpx.stream(
        "POST",
        f"{LLM_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                except json.JSONDecodeError:
                    continue


def get_llm_cache_info() -> dict:
    """获取 LLM 缓存统计。"""
    return _llm_cache.get_info()


def clear_llm_cache():
    """清空 LLM 缓存。"""
    _llm_cache.clear()
    logger.info("LLM 缓存已清空")
