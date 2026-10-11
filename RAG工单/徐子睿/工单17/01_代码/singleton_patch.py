# -*- coding: utf-8 -*-
"""
工单 17 · 优化补丁（草案）：资源池化 + 单例 + 连接池 + 请求限流
——针对「瓶颈一：每请求重复初始化重量级组件」「泄漏点：连接未归还」的修复。

说明：以下为可直接并入 RAGFlow 的 patch 片段（以注释标注接入点）。
"""
from __future__ import annotations

import asyncio
import threading
from functools import lru_cache
from typing import Any


# ============ 修复①：重量级组件全局单例（进程内缓存） ============
class _SingletonRegistry:
    _instances: dict = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, key: str, factory):
        """key 相同则复用同一实例，避免每个请求重新初始化. 线程安全。"""
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    cls._instances[key] = factory()
        return cls._instances[key]


def get_pdf_parser(pdf_cls, **kwargs):
    # 接入点：deepdoc/parser/pdf_parser.py 的 RAGFlowPdfParser 实例化处
    return _SingletonRegistry.get("pdf_parser", lambda: pdf_cls(**kwargs))


def get_rerank_model(llm_bundle_factory, tenant_id, rerank_id):
    # 接入点：api/db/services/dialog_service.py :: get_models()
    #   rerank_mdl = LLMBundle(...) 每请求新建 -> 改走单例
    key = f"rerank:{tenant_id}:{rerank_id}"
    return _SingletonRegistry.get(key, lambda: llm_bundle_factory(tenant_id, rerank_id))


# ============ 修复②：检索/生成链路并发限流（防请求堆积） ============
class RequestLimiter:
    """基于信号量的并发限流：超出并发上限的请求排队，而非无限制抢占资源。"""
    def __init__(self, max_concurrency: int = 16):
        self._sem = asyncio.Semaphore(max_concurrency)

    async def run(self, coro):
        async with self._sem:
            return await coro


# ============ 修复③：连接池正确归还（ES / Redis / MinIO） ============
class PooledClient:
    """
    接入点：确保 DocStore 连接 / redis 客户端 / MinIO 客户端为进程级单例，
    请求处理完不做 close()，而是复用；只在进程退出时释放。
    （RAGFlow 的 settings.docStoreConn / REDIS_CONN 本身是模块级单例——
      排查时重点是确认没有在请求路径里重复 new client 或忘记归还连接。）
    """
    def __init__(self, factory):
        self._factory = factory
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._factory()
        return self._client


# ============ 修复④：解析任务异步化（API Server 不阻塞） ============
# 接入点：api/apps/document_app.py 的上传/解析接口。
#   文档解析应只做「入库 + 投 Redis 队列（queue_tasks）」，立即返回；
#   真正的解析由 Task Executor 消费 —— 这是 RAGFlow 既有设计，
#   泄漏排查时要确认没有在 API 进程里同步触发解析。
async def enqueue_only(queue_product, queue_name, task) -> bool:
    """只投递，不等待解析完成。"""
    return queue_product(queue_name, task)


if __name__ == "__main__":
    # 简单自检：单例只创建一次
    calls = {"n": 0}
    def factory():
        calls["n"] += 1
        return object()
    a = _SingletonRegistry.get("demo", factory)
    b = _SingletonRegistry.get("demo", factory)
    assert a is b and calls["n"] == 1
    print("singleton patch self-test OK")
