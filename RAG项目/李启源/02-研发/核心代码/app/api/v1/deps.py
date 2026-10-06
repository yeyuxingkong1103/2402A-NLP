"""会话/记忆组件的惰性装配。

设计原则：**记忆是可选增强，不是硬依赖**。

Redis 没起、`DATABASE_URL` 没配、或者表还没建，都不能让 `/chat` 挂掉——
它必须退回到今天这个「无状态单轮问答」的行为。所以这里所有 getter 都返回
``None`` 而不是抛异常，调用方用 ``if store is None`` 决定走不走记忆分支。

只探测一次并把结果缓存下来（包括失败）。否则 Redis 没起的时候，
每个请求都要等一次 TCP 连接超时，接口延迟会从 8 秒变成 8 秒 + N 次超时。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_probed = False
_redis_client: Any | None = None
_mysql_client: Any | None = None
_redis_memory: Any | None = None
_session_manager: Any | None = None
_milvus_memory_client: Any | None = None
_milvus_memory: Any | None = None
_memory_manager: Any | None = None
_long_term_memory_error: str | None = None


def _probe() -> None:
    """建立 Redis / MySQL / Milvus 连接，失败就记为不可用。只跑一次。"""
    global _probed, _redis_client, _mysql_client, _redis_memory, _session_manager
    global _milvus_memory_client, _milvus_memory, _memory_manager
    global _long_term_memory_error

    if _probed:
        return

    with _lock:
        if _probed:
            return

        # --- Redis ---
        redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        try:
            import redis as redis_lib

            client = redis_lib.from_url(
                redis_url,
                # decode_responses 必须是 False：RedisMemory.get_session_meta 和
                # SessionManager.get_session 都在手动 .decode('utf-8')，
                # 开了自动解码它们会 AttributeError: 'str' object has no attribute 'decode'。
                decode_responses=False,
                protocol=2,
                socket_connect_timeout=2,
                socket_timeout=3,
            )
            client.ping()
            _redis_client = client
            logger.info("Redis 已连接: %s", redis_url)
        except Exception as exc:
            _redis_client = None
            logger.warning("Redis 不可用，会话记忆功能关闭: %s", exc)

        # --- MySQL ---
        database_url = os.getenv("DATABASE_URL", "").strip()
        if database_url:
            try:
                from app.memory.mysql_client import MySQLClient

                client = MySQLClient(
                    database_url, pool_size=int(os.getenv("DB_POOL_SIZE", "5"))
                )
                if client.ping():
                    _mysql_client = client
                    logger.info("MySQL 已连接")
                else:
                    _mysql_client = None
            except Exception as exc:
                _mysql_client = None
                logger.warning("MySQL 不可用，会话持久化关闭: %s", exc)
        else:
            logger.info("DATABASE_URL 未配置，会话持久化关闭")

        # --- 组装 ---
        if _redis_client is not None:
            from app.memory.redis_memory import RedisMemory

            _redis_memory = RedisMemory(_redis_client)

        if _redis_client is not None and _mysql_client is not None:
            from app.memory.session_manager import SessionManager

            _session_manager = SessionManager(_redis_client, _mysql_client)
            logger.info("会话管理已启用")

        # --- Milvus 长期记忆 ---
        long_term_enabled = os.getenv("LONG_TERM_MEMORY_ENABLED", "true").lower() == "true"
        if long_term_enabled:
            try:
                from app.ingestion.embedding import build_embedding_client_from_env
                from app.memory.milvus_memory import (
                    EmbeddingQueryAdapter,
                    MilvusMemory,
                    MilvusMemoryClient,
                )

                embedding_client = build_embedding_client_from_env()
                _milvus_memory_client = MilvusMemoryClient(
                    uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
                    token=os.getenv("MILVUS_TOKEN") or None,
                    dimension=embedding_client.dimension,
                    collection_prefix=os.getenv("MEMORY_COLLECTION_PREFIX", "rag_v1"),
                )
                _milvus_memory = MilvusMemory(
                    _milvus_memory_client,
                    EmbeddingQueryAdapter(embedding_client),
                )
                logger.info("Milvus 长期记忆已启用")
            except Exception as exc:
                _milvus_memory_client = None
                _milvus_memory = None
                _long_term_memory_error = str(exc)
                logger.warning("Milvus 长期记忆不可用，按无长期记忆处理: %s", exc)
        else:
            _long_term_memory_error = "disabled by LONG_TERM_MEMORY_ENABLED"
            logger.info("Milvus 长期记忆已通过配置关闭")

        if (
            _redis_memory is not None
            and _mysql_client is not None
            and _milvus_memory is not None
        ):
            from app.memory.memory_manager import MemoryManager

            _memory_manager = MemoryManager(
                _redis_memory,
                _milvus_memory,
                _mysql_client,
            )
            logger.info("统一记忆管理已启用")

        _probed = True


def get_redis_memory() -> Any | None:
    """短期对话记忆。只要 Redis 在就可用，不需要 MySQL。"""
    _probe()
    return _redis_memory


def get_session_manager() -> Any | None:
    """会话生命周期管理。需要 Redis + MySQL 同时可用。"""
    _probe()
    return _session_manager


def get_mysql_client() -> Any | None:
    _probe()
    return _mysql_client


def get_milvus_memory() -> Any | None:
    """按用户和租户隔离的 Milvus 长期记忆。"""
    _probe()
    return _milvus_memory


def get_memory_manager() -> Any | None:
    """统一记忆管理器。需要 Redis、MySQL 与 Milvus 同时可用。"""
    _probe()
    return _memory_manager


def memory_status() -> dict[str, Any]:
    """给健康检查用的能力报告。"""
    _probe()
    status = {
        "redis": _redis_client is not None,
        "mysql": _mysql_client is not None,
        "short_term_memory": _redis_memory is not None,
        "session_management": _session_manager is not None,
        "long_term_memory": _milvus_memory is not None,
        "memory_manager": _memory_manager is not None,
    }
    if _long_term_memory_error:
        status["long_term_memory_error"] = _long_term_memory_error
    return status


def reset_for_tests() -> None:
    """清掉缓存的探测结果，让测试能重新装配。"""
    global _probed, _redis_client, _mysql_client, _redis_memory, _session_manager
    global _milvus_memory_client, _milvus_memory, _memory_manager
    global _long_term_memory_error
    with _lock:
        _probed = False
        _redis_client = None
        _mysql_client = None
        _redis_memory = None
        _session_manager = None
        _milvus_memory_client = None
        _milvus_memory = None
        _memory_manager = None
        _long_term_memory_error = None


def resolve_role_id(role_key: str = "customer_service") -> int | None:
    """把角色标识换成数据库自增 id。

    `sessions.role_id` 有外键指向 `roles(id)`，而 `roles` 是自增主键、
    导入顺序不保证，所以不能写死 1。查不到就返回 None，让调用方报错，
    而不是插一个会违反外键的假 id。
    """
    client = get_mysql_client()
    if client is None:
        return None
    row = client.fetchone(
        "SELECT id FROM roles WHERE role_key = %s AND status = 'active' LIMIT 1",
        (role_key,),
    )
    return int(row["id"]) if row else None


def default_user_id() -> int:
    """默认用户。

    接口现在没有鉴权，识别不出调用方是谁，所以统一挂到 schema 里
    预置的测试用户上。等接了鉴权，这里应该换成登录态里的 user_id。
    """
    return int(os.getenv("DEFAULT_USER_ID", "1001"))
