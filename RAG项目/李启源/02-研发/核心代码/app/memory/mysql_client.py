"""MySQL 客户端适配层。

`SessionManager` / `MemoryManager` 期望的是 DB-API 风格的客户端：
``execute(sql, params)`` + ``fetchone(sql, params)``，占位符用 ``%s``，
``fetchone`` 返回 dict。但项目里唯一的数据库访问是 `indexer.SqlAlchemyMetadataStore`，
用的是 SQLAlchemy 的 ``text()`` + 具名参数 ``:name``，两边对不上——
这就是记忆模块虽然写完了却一直没接到 HTTP 层的直接障碍之一。

这里用 PyMySQL 直连补上那个接口。之所以不套 SQLAlchemy：
`SessionManager` 里的 SQL 全是写死的 ``%s``，套过去得逐条改写，
而 PyMySQL 本来就是 `DATABASE_URL` 里 ``mysql+pymysql://`` 的底层驱动，
已经是依赖了，不引入新东西。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)


class MySQLClient:
    """PyMySQL 连接池的薄封装，提供 DB-API 风格接口。

    连接是惰性建立的：没配 ``DATABASE_URL`` 时构造会抛异常，
    调用方应当据此降级到「无持久化」模式，而不是让整个服务起不来。
    """

    def __init__(self, database_url: str, *, pool_size: int = 5) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is empty")

        try:
            import pymysql  # noqa: F401
            from dbutils.pooled_db import PooledDB
        except ImportError:
            # DBUtils 不是硬依赖，没装就退化成「每次新建连接」。
            # 本机自用场景下够了，并发高时再装 DBUtils。
            self._pool = None
            self._config = _parse_mysql_url(database_url)
            self._lock = threading.Lock()
            logger.info("DBUtils 未安装，MySQLClient 使用短连接模式")
            return

        self._config = _parse_mysql_url(database_url)
        self._lock = threading.Lock()
        import pymysql
        from pymysql.cursors import DictCursor

        self._pool = PooledDB(
            creator=pymysql,
            maxconnections=pool_size,
            blocking=True,
            ping=1,  # 取连接时 ping 一次，避免拿到被 MySQL 掐掉的空闲连接
            cursorclass=DictCursor,
            **self._config,
        )
        logger.info("MySQLClient 连接池已建立 (size=%d)", pool_size)

    def _connect(self) -> Any:
        if self._pool is not None:
            return self._pool.connection()
        import pymysql
        from pymysql.cursors import DictCursor

        return pymysql.connect(cursorclass=DictCursor, **self._config)

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        """执行写语句并提交，返回受影响行数。"""
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                affected = cursor.execute(sql, params or ())
            conn.commit()
            return affected
        finally:
            conn.close()

    def fetchone(self, sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or ())
                return cursor.fetchone()
        finally:
            conn.close()

    def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or ())
                return list(cursor.fetchall())
        finally:
            conn.close()

    def ping(self) -> bool:
        try:
            self.fetchone("SELECT 1 AS ok")
            return True
        except Exception as exc:  # pragma: no cover - 只用于健康检查
            logger.warning("MySQL ping 失败: %s", exc)
            return False


def _parse_mysql_url(url: str) -> dict[str, Any]:
    """把 SQLAlchemy 风格的 URL 拆成 PyMySQL 的连接参数。

    接受 ``mysql+pymysql://user:pass@host:port/db?charset=utf8mb4``，
    也接受不带 ``+pymysql`` 的形式。用户名/密码里的百分号转义要还原，
    否则密码含 ``@`` 或 ``#`` 时会连不上。
    """
    parsed = urlparse(url)
    if not parsed.scheme.startswith("mysql"):
        raise ValueError(f"不是 MySQL 连接串: {parsed.scheme}")

    charset = "utf8mb4"
    for pair in (parsed.query or "").split("&"):
        if pair.startswith("charset="):
            charset = pair.split("=", 1)[1] or charset

    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": (parsed.path or "/").lstrip("/"),
        "charset": charset,
        "autocommit": False,
    }
