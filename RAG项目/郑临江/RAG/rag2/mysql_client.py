# -*- coding: utf-8 -*-
"""MySQL 数据库调用模块（pymysql）。

基于 pymysql 封装常用增删改查，返回字典行（DictCursor），方便在 RAG 流程中
读写元数据 / 检索结果 / 配置表等结构化数据。

实现要点：

1. **懒加载 / 懒连接**：顶层不 ``import pymysql``，首次真正执行 SQL 时才导入并连接；
2. **字典行**：查询结果直接是 ``list[dict]``，列名即键，无需手工按列号取值；
3. **占位符**：统一使用 pymysql 的 ``%s`` 占位符（不是 ``?``），参数自动转义防注入；
4. **连接管理**：支持 ``with MySQLClient(...) as db``、自动重连（``ping``）、
   ``transaction()`` 事务上下文，及 `threading` 锁保证同一连接串行使用；
5. **便捷函数**：模块级 ``connect`` / ``query`` / ``execute`` 等一键调用。

运行环境：

    使用已安装 pymysql 的 rags_ 环境（或 base）运行，例如：

        D:/an/envs/rags_/python.exe -c "from mysql_client import query; print(query('SELECT 1'))"

用法示例（详见同目录 README.md）：

    from mysql_client import MySQLClient

    db = MySQLClient(host="localhost", user="root", password="xxx", database="mydb")
    rows = db.query("SELECT id, name FROM user WHERE age > %s", (18,))   # -> [{'id':..,'name':..}]
    n = db.execute("UPDATE user SET name=%s WHERE id=%s", ("张三", 1))    # -> 影响行数
    new_id = db.insert("user", {"name": "李四", "age": 20})               # -> 自增主键

    with MySQLClient(host="localhost", user="root", password="xxx", database="mydb") as db:
        with db.transaction():
            db.execute("INSERT INTO log(msg) VALUES (%s)", ("hello",))
            db.execute("INSERT INTO log(msg) VALUES (%s)", ("world",))
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

logger = logging.getLogger("rag2.mysql_client")


@dataclass
class MySQLConfig:
    """MySQL 连接配置。"""

    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str | None = None      # 缺省不选库，可稍后 USE 或用全限定表名
    charset: str = "utf8mb4"
    connect_timeout: int = 10
    read_timeout: int = 30
    write_timeout: int = 30
    autocommit: bool = True

    @classmethod
    def from_env(cls, prefix: str = "MYSQL_") -> "MySQLConfig":
        """从环境变量读取配置（``<prefix>HOST/PORT/USER/PASSWORD/DATABASE/CHARSET``）。"""
        get = lambda k, d: os.environ.get(prefix + k, d)  # noqa: E731
        return cls(
            host=get("HOST", "localhost"),
            port=int(get("PORT", 3306)),
            user=get("USER", "root"),
            password=get("PASSWORD", ""),
            database=get("DATABASE", None) or None,
            charset=get("CHARSET", "utf8mb4"),
        )

    def as_dict(self) -> dict:
        return asdict(self)

    def masked(self) -> dict:
        """返回脱敏后的配置（密码打码），便于日志打印。"""
        d = self.as_dict()
        d["password"] = "***" if d["password"] else ""
        return d


class MySQLClient:
    """MySQL 客户端封装（懒连接，字典行，可作上下文管理器）。"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        charset: str | None = None,
        config: MySQLConfig | dict | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化（不立即连接）。

        参数：
            host/port/user/password/database/charset: 连接参数，缺省取 config 或默认值。
            config:   MySQLConfig 对象或 dict，与显式参数合并（显式参数优先）。
            **kwargs: 其余透传给 pymysql.connect（如 ssl、read_timeout 等）。

        例：
            db = MySQLClient(config=MySQLConfig.from_env())
            db = MySQLClient(host="127.0.0.1", user="root", password="p", database="db")
        """
        base = dict(config.as_dict()) if isinstance(config, MySQLConfig) else dict(config or {})
        base.update(kwargs)
        if host is not None:
            base["host"] = host
        if port is not None:
            base["port"] = int(port)
        if user is not None:
            base["user"] = user
        if password is not None:
            base["password"] = password
        if database is not None:
            base["database"] = database
        if charset is not None:
            base["charset"] = charset

        known = {f.name for f in MySQLConfig.__dataclass_fields__.values()}
        cfg_kwargs = {k: v for k, v in base.items() if k in known}
        self.cfg = MySQLConfig(**cfg_kwargs)
        self.extra_connect = {k: v for k, v in base.items() if k not in known}

        self._conn: Any = None
        self.lastrowid: int | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 连接
    def _connect(self) -> Any:
        import pymysql
        import pymysql.cursors

        conn = pymysql.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            user=self.cfg.user,
            password=self.cfg.password,
            database=self.cfg.database,
            charset=self.cfg.charset,
            connect_timeout=self.cfg.connect_timeout,
            read_timeout=self.cfg.read_timeout,
            write_timeout=self.cfg.write_timeout,
            autocommit=self.cfg.autocommit,
            cursorclass=pymysql.cursors.DictCursor,
            **self.extra_connect,
        )
        logger.info("连接 MySQL：%s:%s/%s", self.cfg.host, self.cfg.port, self.cfg.database or "-")
        return conn

    def connect(self) -> "MySQLClient":
        """建立连接（幂等），返回自身便于链式调用。"""
        with self._lock:
            if self._conn is None:
                try:
                    self._conn = self._connect()
                except ImportError as exc:  # pragma: no cover - 依赖缺失提示
                    raise ImportError(
                        "未找到 pymysql，请使用 rags_ 环境运行：D:/an/envs/rags_/python.exe。"
                    ) from exc
            return self

    @property
    def conn(self) -> Any:
        """懒加载连接对象。"""
        if self._conn is None:
            self.connect()
        return self._conn

    def close(self) -> None:
        """关闭连接。"""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001 - 忽略关闭异常
                    pass
                self._conn = None

    def ping(self, reconnect: bool = True) -> bool:
        """探测连接是否存活，可选自动重连。"""
        with self._lock:
            if self._conn is None:
                return False
            try:
                self._conn.ping(reconnect=reconnect)
                return True
            except Exception:
                return False

    def is_connected(self) -> bool:
        return self._conn is not None and self.ping(reconnect=False)

    # ------------------------------------------------------------------ 查询
    @staticmethod
    def _params(params: Any) -> tuple:
        """把参数统一成元组，交给 pymysql 做参数化查询。

        SQL 里的 ``%s`` 占位符由 pymysql 自动转义，能防 SQL 注入——
        所以永远不要把用户输入直接拼进 SQL 字符串，而是走这里。
        """
        if params is None:
            return ()
        if isinstance(params, (list, tuple)):
            return tuple(params)
        return (params,)

    @staticmethod
    def _q(name: str) -> str:
        """给标识符（表名/列名）加反引号，避免关键字/特殊字符冲突。"""
        name = str(name).replace("`", "``")
        return f"`{name}`"

    def query(self, sql: str, params: Any = None) -> list[dict]:
        """执行 SELECT/SHOW 等查询，返回字典行列表。

        参数：
            sql:    SQL 语句，占位符用 ``%s``。
            params: 参数元组/列表/单值。

        返回：
            ``list[dict]``，每行一个字典（列名 -> 值）。
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(sql, self._params(params))
                return list(cur.fetchall())
            finally:
                cur.close()

    def query_one(self, sql: str, params: Any = None) -> dict | None:
        """执行查询并返回第一行（字典）；无结果返回 None。"""
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(sql, self._params(params))
                return cur.fetchone()
            finally:
                cur.close()

    def query_scalar(self, sql: str, params: Any = None) -> Any:
        """执行查询并返回第一行第一列的值；无结果返回 None。"""
        row = self.query_one(sql, params)
        if row is None:
            return None
        return next(iter(row.values()))

    # ------------------------------------------------------------------ 写入
    def execute(self, sql: str, params: Any = None) -> int:
        """执行 INSERT/UPDATE/DELETE 等语句，返回影响行数。

        自增主键写入后可通过 ``db.lastrowid`` 取得。
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(sql, self._params(params))
                self.lastrowid = cur.lastrowid
                return int(cur.rowcount or 0)
            finally:
                cur.close()

    def executemany(self, sql: str, seq_params: Iterable[Sequence]) -> int:
        """批量执行同一语句（每行一组参数），返回总影响行数。"""
        params_list = [self._params(p) for p in seq_params]
        if not params_list:
            return 0
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.executemany(sql, params_list)
                self.lastrowid = cur.lastrowid
                return int(cur.rowcount or 0)
            finally:
                cur.close()

    # ------------------------------------------------------------------ 便捷写入
    def insert(self, table: str, data: dict) -> int:
        """插入一行（dict 的键为列名），返回自增主键（无自增时返回 0）。"""
        if not data:
            raise ValueError("insert 数据不能为空")
        cols = list(data.keys())
        placeholders = ", ".join(["%s"] * len(cols))
        sql = (f"INSERT INTO {self._q(table)} "
               f"({', '.join(self._q(c) for c in cols)}) VALUES ({placeholders})")
        self.execute(sql, [data[c] for c in cols])
        return int(self.lastrowid or 0)

    def insert_many(self, table: str, rows: Iterable[dict]) -> int:
        """批量插入多行（列名取自首行），返回影响行数。"""
        rows = list(rows)
        if not rows:
            return 0
        cols = list(rows[0].keys())
        placeholders = ", ".join(["%s"] * len(cols))
        sql = (f"INSERT INTO {self._q(table)} "
               f"({', '.join(self._q(c) for c in cols)}) VALUES ({placeholders})")
        return self.executemany(sql, [[r.get(c) for c in cols] for r in rows])

    # ------------------------------------------------------------------ 元数据
    def version(self) -> str:
        """返回 MySQL 服务器版本。"""
        return str(self.query_scalar("SELECT VERSION()"))

    def current_db(self) -> str | None:
        """返回当前选中的数据库名（未选库时返回 None）。"""
        return self.query_scalar("SELECT DATABASE()")

    def tables(self) -> list[str]:
        """返回当前库的全部表名。"""
        rows = self.query("SHOW TABLES")
        return [next(iter(r.values())) for r in rows]

    def table_exists(self, table: str) -> bool:
        """判断表是否存在。"""
        return len(self.query("SHOW TABLES LIKE %s", (table,))) > 0

    def columns(self, table: str) -> list[dict]:
        """返回表的列信息（等价于 SHOW COLUMNS）。"""
        return self.query(f"SHOW COLUMNS FROM {self._q(table)}")

    # ------------------------------------------------------------------ 事务
    @contextmanager
    def transaction(self):
        """事务上下文：块内语句一并提交，异常回滚。

        例：
            with db.transaction():
                db.execute("INSERT ...")
                db.execute("INSERT ...")
        """
        conn = self.conn
        with self._lock:
            was_auto = bool(conn.get_autocommit())
            if was_auto:
                conn.autocommit(False)
        try:
            yield self
            with self._lock:
                conn.commit()
        except Exception:
            with self._lock:
                conn.rollback()
            raise
        finally:
            with self._lock:
                if was_auto:
                    conn.autocommit(True)

    # ------------------------------------------------------------------ 上下文
    def __enter__(self) -> "MySQLClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - 调试打印
        return (f"MySQLClient({self.cfg.host}:{self.cfg.port}/{self.cfg.database or '-'}, "
                f"connected={self._conn is not None})")


# ---------------------------------------------------------------------- 便捷函数
def connect(**kwargs: Any) -> MySQLClient:
    """便捷函数：创建并立即连接 MySQLClient（等价于 ``MySQLClient(**kw).connect()``）。"""
    return MySQLClient(**kwargs).connect()


def query(sql: str, params: Any = None, **conn: Any) -> list[dict]:
    """便捷函数：一次性连接执行查询并关闭，返回字典行列表。"""
    with connect(**conn) as db:
        return db.query(sql, params)


def query_one(sql: str, params: Any = None, **conn: Any) -> dict | None:
    """便捷函数：一次性连接执行查询并返回第一行。"""
    with connect(**conn) as db:
        return db.query_one(sql, params)


def query_scalar(sql: str, params: Any = None, **conn: Any) -> Any:
    """便捷函数：一次性连接执行查询并返回第一行第一列。"""
    with connect(**conn) as db:
        return db.query_scalar(sql, params)


def execute(sql: str, params: Any = None, **conn: Any) -> int:
    """便捷函数：一次性连接执行写入并关闭，返回影响行数。"""
    with connect(**conn) as db:
        return db.execute(sql, params)


__all__ = [
    "MySQLConfig",
    "MySQLClient",
    "connect",
    "query",
    "query_one",
    "query_scalar",
    "execute",
]


if __name__ == "__main__":  # pragma: no cover - 无 MySQL 时仅演示 SQL 构建
    cfg = MySQLConfig.from_env()
    print("配置（脱敏）:", cfg.masked())

    db = MySQLClient(config=cfg)
    # 纯本地演示：构建一条 INSERT 语句，不依赖真实连接
    print("INSERT 示例:", f"INSERT INTO {db._q('user')} "
          f"({', '.join(db._q(c) for c in ['name', 'age'])}) VALUES (%s, %s)")

    if cfg.database:
        try:
            with db:
                print("服务器版本:", db.version())
                print("表列表:", db.tables())
        except Exception as exc:
            print("连接失败（请检查 MYSQL_* 环境变量）：", type(exc).__name__, exc)
    else:
        print("\n未设置 MYSQL_DATABASE，跳过真实连接。")
        print("示例：MYSQL_DATABASE=mydb MYSQL_PASSWORD=xxx "
              "D:/an/envs/rags_/python.exe mysql_client.py")
