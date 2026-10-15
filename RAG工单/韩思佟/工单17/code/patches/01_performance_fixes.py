# -*- coding: utf-8 -*-
"""
工单17 —— 性能与资源补丁合集（P1 ~ P5）

每个补丁都以「原代码 → 问题 → 补丁后代码」的形式给出，可直接对照 RAGFlow 0.23.1 源码落地。
落地位置已在每段注释里标明。

P1 线程池单例       → common/misc_utils.py
P2 模型单例         → api/db/services/tenant_llm_service.py + deepdoc 解析器
P3 连接池           → api/db/db_models.py + rag/utils/redis_conn.py
P4 请求限流         → api/apps/*.py 问答入口
P5 资源释放         → 中间件 / 请求结束钩子
"""
from __future__ import annotations

# ============================================================================
# P1  线程池单例（修复：每次调用都新建 ThreadPoolExecutor）
# ----------------------------------------------------------------------------
# 【原代码】 common/misc_utils.py:116
#   def _thread_pool_executor():
#       max_workers_env = os.getenv("THREAD_POOL_MAX_WORKERS", "128")
#       ...
#       return ThreadPoolExecutor(max_workers=max_workers)
#   async def thread_pool_exec(func, *args, **kwargs):
#       loop = asyncio.get_running_loop()
#       return await loop.run_in_executor(_thread_pool_executor(), func)
#
# 【问题】每次调用新建 128 线程的池，高并发下线程数量爆炸，上下文切换吃掉大量 CPU。
# ============================================================================
from concurrent.futures import ThreadPoolExecutor
import asyncio
import functools
import os
import atexit

_thread_pool: ThreadPoolExecutor | None = None


def _thread_pool_executor() -> ThreadPoolExecutor:
    """P1：全局单例线程池（只创建一次，进程退出时优雅关闭）"""
    global _thread_pool
    if _thread_pool is None:
        max_workers = int(os.getenv("THREAD_POOL_MAX_WORKERS", "64"))
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers,
                                          thread_name_prefix="rag_tp")
    return _thread_pool


@atexit.register
def _shutdown_thread_pool():
    global _thread_pool
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=False)
        _thread_pool = None


async def thread_pool_exec(func, *args, **kwargs):
    """P1：复用单例线程池"""
    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(_thread_pool_executor(), func)
    return await loop.run_in_executor(_thread_pool_executor(), func, *args)


# ============================================================================
# P2  模型单例（修复：LLMBundle / model_instance 每次新建客户端）
# ----------------------------------------------------------------------------
# 【原代码】 api/db/services/tenant_llm_service.py:378
#   self.mdl = TenantLLMService.model_instance(tenant_id, llm_type, llm_name, ...)
#   而 model_instance() 内部对每种 LLMType 直接 new 一个客户端，无缓存。
# 【问题】每个请求都可能重新初始化重量级客户端，GPU/内存被反复占用且不及时释放。
# ============================================================================
from functools import lru_cache
from threading import Lock

_model_cache = {}
_model_lock = Lock()


def model_instance_cached(tenant_id: str, llm_type: str, llm_name: str,
                          lang: str | None = None, factory=None):
    """
    P2：进程内模型单例。
    key = (tenant_id, llm_type, llm_name, lang)；命中直接返回，未命中才新建并缓存。
    注意：切换模型配置时需要清缓存（见 invalidate_model_cache）。
    """
    key = (tenant_id, llm_type, llm_name, lang)
    mdl = _model_cache.get(key)
    if mdl is not None:
        return mdl
    with _model_lock:
        mdl = _model_cache.get(key)
        if mdl is None:
            mdl = factory(tenant_id, llm_type, llm_name, lang) if factory else None
            if mdl is not None:
                _model_cache[key] = mdl
    return mdl


def invalidate_model_cache(tenant_id: str | None = None):
    """模型配置变更后调用，避免用到旧实例"""
    global _model_cache
    if tenant_id is None:
        _model_cache.clear()
    else:
        for k in [k for k in _model_cache if k[0] == tenant_id]:
            _model_cache.pop(k, None)


# ---- deepdoc 解析器单例（同样的问题）
_parser_cache = {}
_parser_lock = Lock()


def get_pdf_parser(parser_cls):
    """P2 补充：deepdoc 解析器按类做单例，避免 OCR/Layout/TSR 反复初始化"""
    key = parser_cls.__name__
    obj = _parser_cache.get(key)
    if obj is None:
        with _parser_lock:
            obj = _parser_cache.get(key)
            if obj is None:
                obj = parser_cls()
                _parser_cache[key] = obj
    return obj


# ============================================================================
# P3  连接池（修复：DB/向量库/Redis 未显式配置池上限）
# ----------------------------------------------------------------------------
# 【原代码】 api/db/db_models.py:396
#   pool_config = {'max_retries': 5, 'retry_delay': 1}
#   （没有 max_connections / stale_timeout，走 playhouse 默认值）
# 【原代码】 rag/utils/redis_conn.py:125
#   self.REDIS = redis.StrictRedis(**conn_params)   # 未显式传 ConnectionPool
# ============================================================================
POOL_CONFIG = {
    # Peewee playhouse.pool 参数
    "max_connections": int(os.getenv("DB_MAX_CONNECTIONS", "32")),
    "stale_timeout": int(os.getenv("DB_STALE_TIMEOUT", "300")),
    "timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
    "max_retries": 5,
    "retry_delay": 1,
}

REDIS_POOL_KWARGS = {
    "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "64")),
    "health_check_interval": 30,
    "socket_keepalive": True,
}


def build_redis_client(host: str, port: int, db: int, password=None):
    """P3：显式使用 ConnectionPool 的 Redis 客户端"""
    import redis
    pool = redis.ConnectionPool(host=host, port=port, db=db,
                                password=password, decode_responses=True,
                                **REDIS_POOL_KWARGS)
    return redis.StrictRedis(connection_pool=pool)


# ============================================================================
# P4  请求限流（修复：无队列/限流，请求堆积）
# ----------------------------------------------------------------------------
# 【思路】给问答接口加信号量 + 排队超时；超出并发上限的请求直接返回 429，
#        避免线程被长时间占满导致 P95 飙升。
#        复用仓库里已有的 Lua 令牌桶：rag/utils/redis_conn.py LUA_TOKEN_BUCKET
# ============================================================================
_chat_sem: asyncio.Semaphore | None = None


def get_chat_semaphore() -> asyncio.Semaphore:
    global _chat_sem
    if _chat_sem is None:
        _chat_sem = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_CHATS", "10")))
    return _chat_sem


async def with_chat_limit(coro_factory, queue_timeout: float | None = None):
    """
    P4：限流包装
      coro_factory: async callable，真正处理请求的协程工厂
    超时直接抛 asyncio.TimeoutError，调用方返回 429。
    """
    qto = queue_timeout or float(os.getenv("CHAT_QUEUE_TIMEOUT", "30"))
    sem = get_chat_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=qto)
    except asyncio.TimeoutError:
        raise
    try:
        return await coro_factory()
    finally:
        sem.release()


# ============================================================================
# P5  资源释放（修复：无显式张量/中间对象释放，内存缓慢上涨）
# ----------------------------------------------------------------------------
# 【现状】全仓库没有 torch.cuda.empty_cache()；只在 deepdoc/vision/recognizer.py:410
#        有 del ort_sess + gc.collect()。
# ============================================================================
import gc


def release_resources(force: bool = False, every_n_requests: int = 16):
    """
    P5：请求结束后的资源释放钩子
      - 默认每 N 个请求做一次 gc.collect()（太频繁会拖慢响应）
      - 有 GPU 时显式 empty_cache()
    """
    release_resources._n = getattr(release_resources, "_n", 0) + 1
    if not force and release_resources._n % every_n_requests != 0:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                            # noqa: BLE001
        pass
    gc.collect()


# ============================================================================
# 使用说明（把下面 5 行挂到对应位置即可）
#   1) P1：把 common/misc_utils.py 的 _thread_pool_executor / thread_pool_exec 替换为上面的实现
#   2) P2：tenant_llm_service.model_instance 外部包一层 model_instance_cached
#          deepdoc 解析器用 get_pdf_parser(RAGFlowPdfParser) 取实例
#   3) P3：BaseDataBase 的 pool_config 改为 POOL_CONFIG；RedisDB.__open__ 用 build_redis_client
#   4) P4：问答接口入口改为 await with_chat_limit(lambda: handle_chat(...))
#   5) P5：在 @app.after_request / teardown_request 里调用 release_resources()
# ============================================================================
