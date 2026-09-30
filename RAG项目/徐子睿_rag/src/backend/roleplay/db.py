# -*- coding: utf-8 -*-
"""roleplay/db.py —— SQLite 连接、建表与内置角色初始化。

在链路中的位置：
    roleplay 包的数据层底座：roles.py（角色）、sessions.py（用户/会话/消息）
    都通过本文件的 _db() 拿连接。

四张表：
    users     用户
    roles     角色卡（config_json 存整份角色配置的 JSON）
    sessions  会话（一个用户在某角色下的一段连续对话）
    messages  消息（speaker 区分 user/assistant）

三个并发相关的处理写在下面的 docstring 里：check_same_thread=False、
row_factory、WAL 日志模式 —— 它们共同保证多人同时聊天时不会互相卡住。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from .config import DEFAULT_ROLES, ROLEPLAY_DB

_DB_LOCK = threading.RLock()  # SQLite 连接是共享的，写操作必须串行
_DB: sqlite3.Connection | None = None

def _now() -> str:
    """取当前时间字符串（统一格式，避免各处各写一套 strftime）。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _db() -> sqlite3.Connection:
    """获取 SQLite 连接（首次调用时建库建表、灌入内置角色）。

    返回：
        进程级共享的 sqlite3.Connection。

    建表说明（四张表，用 TEXT 主键存 uuid，时间用 TEXT 存字符串）：
        users     —— 用户。id/name/created_at
        roles     —— 角色卡。config_json 存整份角色配置的 JSON，
                     这样以后给角色加字段不用改表结构；id/name/description 另存列便于列表查询
        sessions  —— 会话。把 user_id + role_id 组合起来，一个用户在同一角色下可以有多个会话；
                     idx_sessions_user 索引按 (user_id, updated_at DESC) 建，
                     正好匹配"列出我的会话、最近的在最前"这个查询
        messages  —— 消息。speaker 区分 user/assistant；
                     idx_messages_session 索引按 (session_id, created_at) 建，匹配按时间读历史

    三个并发相关的处理：
        check_same_thread=False —— 允许连接跨线程使用（FastAPI 是多线程的），
                                   并发安全由 _DB_LOCK 和应用层的串行写来保证
        row_factory = sqlite3.Row —— 让查询结果能按列名取值，而不是只能按数字下标
        journal_mode=WAL —— 写日志模式。读操作不再阻塞写、写也不阻塞读，
                            多人同时聊天时不会互相卡住

    INSERT OR IGNORE 灌内置角色：
        已存在的角色（用户改过的）不会被覆盖，只有缺失的才插入 ——
        这样升级代码新增内置角色时，不会把用户的自定义修改冲掉。
    """
    global _DB
    with _DB_LOCK:
        if _DB is None:
            ROLEPLAY_DB.parent.mkdir(parents=True, exist_ok=True)  # 首次运行时 data 目录可能不存在
            _DB = sqlite3.connect(str(ROLEPLAY_DB), check_same_thread=False)
            _DB.row_factory = sqlite3.Row
            _DB.execute("PRAGMA journal_mode=WAL")
            _DB.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS roles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    role_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
                """
            )
            for role in DEFAULT_ROLES:
                _DB.execute(
                    "INSERT OR IGNORE INTO roles(id,name,description,config_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (role["id"], role["name"], role["description"], json.dumps(role, ensure_ascii=False), _now(), _now()),
                )
            _DB.commit()
        return _DB
