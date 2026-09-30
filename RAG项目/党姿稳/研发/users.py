"""
users.py — 用户注册与登录（MySQL，连不上自动降级 SQLite）

只填一个姓名即可，没有密码：

    首次输入姓名 → 新建用户并生成 user_id
    再次输入同名 → 直接返回既有 user_id

所以老用户开新对话不需要重复注册，而 user_id 就是注册 ID，
短期记忆与长期记忆都按它归档。

存储后端：生产模式（LOCAL_MODE=False）连 MySQL，用户信息落在 users 表；
本地模式或 MySQL 不可用时降级到 storage/users.db，保证断网也能跑。
对外接口与后端无关，其他模块不用改。
"""

from __future__ import annotations

import secrets
import time
from contextlib import contextmanager

import config

# MySQL 与 SQLite 的字段类型不同，各留一份建表语句，列定义保持一致
_MYSQL_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS users ("
    "user_id VARCHAR(64) PRIMARY KEY, "
    "name VARCHAR(32) UNIQUE NOT NULL, "
    "created_at BIGINT NOT NULL"
    ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
)
_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    created_at INTEGER NOT NULL
)
"""

# 姓名长度上限，防止把整段文字填进输入框
_MAX_NAME = 32


def _warn(message: str) -> None:
    """降级时留一条日志，避免生产模式下静默写进了 SQLite 没人发现。"""
    try:
        from loguru import logger

        logger.warning(message)
    except Exception:
        pass


def _connect() -> tuple[str, object]:
    """新建连接，返回 (后端名, 连接对象)。

    生产模式优先连 MySQL；没装 pymysql 或连不上时降级到 SQLite。
    连接不缓存：SQLite 连接不能跨线程共享，而检索是并行的。
    """
    if not config.LOCAL_MODE:
        try:
            import pymysql

            conn = pymysql.connect(
                host=config.MYSQL_HOST,
                port=config.MYSQL_PORT,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DB,
                charset="utf8mb4",
                connect_timeout=5,
                cursorclass=pymysql.cursors.DictCursor,
            )
            return "mysql", conn
        except Exception as exc:
            _warn(f"MySQL 不可用（{exc}），用户信息降级写入本地 SQLite。")

    import sqlite3

    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.STORAGE_DIR / "users.db", timeout=10)
    conn.row_factory = sqlite3.Row
    return "sqlite", conn


def _run(backend: str, conn, statement: str, params: tuple = ()):
    """执行一条语句并返回游标。

    两个驱动的差异都收在这里：SQLite 用 ? 占位、连接本身可当游标用；
    MySQL 用 %s 占位，且 execute() 返回的是影响行数，不能链式取结果。
    """
    if backend == "mysql":
        cursor = conn.cursor()
        cursor.execute(statement.replace("?", "%s"), params)
        return cursor
    return conn.execute(statement, params)


@contextmanager
def _session(conn):
    """连接的统一上下文：正常退出提交，出错回滚，最后关闭。

    两个驱动的连接上下文不能共用写法：pymysql 的 __exit__ 只 close 不提交，
    SQLite 的只提交不 close，叠用还会报 "Already closed"，所以差异收在这里。
    """
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    else:
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def init_db() -> None:
    """建表，可重复调用。"""
    backend, conn = _connect()
    with _session(conn):
        if backend == "mysql":
            with conn.cursor() as cursor:
                cursor.execute(_MYSQL_SCHEMA)
        else:
            conn.executescript(_SQLITE_SCHEMA)


def login_or_register(name: str) -> tuple[str, bool]:
    """按姓名登录，姓名不存在就注册。

    返回 (user_id, 是否新建)。姓名为空或过长会抛 ValueError。
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("请先填写姓名")
    if len(name) > _MAX_NAME:
        raise ValueError(f"姓名太长了，最多 {_MAX_NAME} 个字符")

    init_db()
    backend, conn = _connect()
    with _session(conn):
        row = _run(
            backend, conn, "SELECT user_id FROM users WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            return str(row["user_id"]), False

        user_id = "u_" + secrets.token_hex(4)
        try:
            _run(
                backend,
                conn,
                "INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
                (user_id, name, int(time.time())),
            )
        except Exception:
            # 两个人同时注册同名时靠 UNIQUE 兜底，重查一次拿已有的那条；
            # 重查也失败说明不是重名冲突，异常照旧抛出
            row = _run(
                backend, conn, "SELECT user_id FROM users WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                raise
            return str(row["user_id"]), False
        return user_id, True


def get(user_id: str) -> dict | None:
    """按 user_id 取用户，找不到返回 None。"""
    if not user_id:
        return None
    init_db()
    backend, conn = _connect()
    with _session(conn):
        row = _run(
            backend, conn, "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def list_users() -> list[dict]:
    """按注册时间倒序列出全部用户，供管理接口使用。"""
    init_db()
    backend, conn = _connect()
    with _session(conn):
        rows = _run(backend, conn, "SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    import shutil
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    config.STORAGE_DIR = tmp
    backend, _ = _connect()
    # 姓名每次现取：库里已有同名用户时，断言"首次登录应新建"就会假失败
    name = f"自检{secrets.token_hex(3)}"
    made: list[str] = []
    try:
        before = len(list_users())
        uid1, created1 = login_or_register(name)
        made.append(uid1)
        uid2, created2 = login_or_register(name)

        assert created1, "首次登录应该新建用户"
        assert not created2, "同名用户应该直接登录而不是重复注册"
        assert uid1 == uid2, "同名用户拿到的 user_id 应该一致"
        assert get(uid1)["name"] == name
        assert len(list_users()) == before + 1, "新用户应该出现在列表里"

        for bad in ("", "   ", "x" * 40):
            try:
                login_or_register(bad)
            except ValueError:
                continue
            raise AssertionError(f"非法姓名 {bad!r} 应该被拒绝")

        print(f"users 自检通过（backend={backend}）：{uid1}")
    finally:
        # 自检数据不留库，免得 MySQL 里攒一堆"自检xxx"
        try:
            cleanup = _connect()[1]
            with _session(cleanup):
                for uid in made:
                    _run(backend, cleanup, "DELETE FROM users WHERE user_id = ?", (uid,))
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
