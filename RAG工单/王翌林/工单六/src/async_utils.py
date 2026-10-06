# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
src/async_utils.py —— 工单二异步/批量并发工具

能力：
  1. encode_batch_async：批量嵌入异步化（分批 + 线程池并行，阻塞型 encode 不卡事件循环）
  2. retrieve_async / generate_async：把同步检索与 LLM 调用包装为协程，供 FastAPI async 端点使用
  3. gather_with_semaphore：并发限速，防止打爆 GPU / LLM API
"""
import asyncio
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger

DEFAULT_CONCURRENCY = 4       # 默认并发批数（人工智能NLP-RAG-基于PDF文档的问答系统优化）
DEFAULT_BATCH_SIZE = 64       # 每批文本数


async def encode_batch_async(embedder, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE,
                             max_concurrency: int = DEFAULT_CONCURRENCY) -> np.ndarray:
    """工单二批量嵌入异步化（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    文本切批 → asyncio.gather 并行（线程池执行阻塞 encode）→ 按原顺序拼接向量"""
    loop = asyncio.get_running_loop()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)] or [[]]
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(batch):
        async with sem:
            return await loop.run_in_executor(
                None, partial(embedder.encode, batch, batch_size=batch_size,
                              show_progress_bar=False))

    t0 = time.time()
    results = await asyncio.gather(*[_one(b) for b in batches])
    arrs = [np.asarray(r, dtype="float32") for r in results]
    vecs = np.vstack(arrs) if arrs else np.zeros((0, 1), dtype="float32")
    logger.info(f"批量嵌入完成: {len(texts)} 条 / {len(batches)} 批 / {max_concurrency} 并发, "
                f"{(time.time() - t0) * 1000:.0f}ms")
    return vecs


async def run_cpu_async(fn: Callable, *args, **kwargs) -> Any:
    """把同步阻塞函数放入线程池执行（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


async def retrieve_async(retriever, query: str, top_k: int = 5) -> Dict[str, Any]:
    """异步检索包装（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    return await run_cpu_async(retriever.retrieve, query, top_k=top_k)


async def generate_async(chat_fn: Callable, messages: List[Dict], **kwargs) -> Dict[str, Any]:
    """异步 LLM 生成包装（OpenAI 客户端为同步实现，人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    return await run_cpu_async(chat_fn, messages=messages, **kwargs)


async def gather_with_semaphore(coros: List, max_concurrency: int = DEFAULT_CONCURRENCY) -> List:
    """并发限速聚合（防止 GPU / LLM API 过载，人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    sem = asyncio.Semaphore(max_concurrency)

    async def _wrap(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_wrap(c) for c in coros])
