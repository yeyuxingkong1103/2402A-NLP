# -*- coding: utf-8 -*-
"""MySQL 持久化层：角色表 + 每个用户最近 N 条对话。

在线阶段的两块 MySQL 数据都收敛到这里：

1. **角色表（roles）**——角色定义以 MySQL 为唯一事实来源：
   启动 Web 端时从 MySQL 读取角色即可确定「有哪些角色」；
   管理员后台的新增 / 编辑 / 删除也直接落 MySQL。
   首次启动（roles 表为空）时用 ``roles.yaml`` / 内置默认角色播种一次。

2. **用户最近对话（user_messages）**——每个用户只保留最近 ``recent_n`` 条
   消息（默认 10），每写入一轮自动裁剪掉更早的记录，实现「每个用户最后 10 条」。

   建表 / 建库都在 ``ensure_schema()`` 里幂等完成，无需手工初始化数据库。

    from rag2.mysql_store import MySQLStore

    store = MySQLStore(host="127.0.0.1", user="root", password="***", database="rag2")
    store.ensure_schema()
    roles = store.load_roles()
    store.append_messages("alice", [{"role": "user", "content": "你好"}, ...])
"""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from .logging_config import get_logger
from .mysql_client import MySQLClient
from .roles import Role, role_from_dict

logger = get_logger("mysql_store")

# 每个用户保留的最近消息条数（「最后 10 条对话数据」）
DEFAULT_RECENT_N = 10

_ROLES_DDL = """
CREATE TABLE IF NOT EXISTS roles (
    role_id       VARCHAR(64)  NOT NULL,
    name          VARCHAR(128) NOT NULL,
    system_prompt TEXT         NOT NULL,
    description   VARCHAR(512) NOT NULL DEFAULT '',
    avatar        VARCHAR(16)  NOT NULL DEFAULT '🙂',
    followups     JSON         NULL,
    updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (role_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_USER_MESSAGES_DDL = """
CREATE TABLE IF NOT EXISTS user_messages (
    id         BIGINT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(128) NOT NULL,
    role       VARCHAR(16)  NOT NULL,
    content    TEXT         NULL,
    role_id    VARCHAR(64)  NOT NULL DEFAULT '',
    session_id VARCHAR(64)  NOT NULL DEFAULT '',
    ts         DOUBLE       NOT NULL DEFAULT 0,
    created_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (username, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

# 裁剪到每个用户最近 recent_n 条：删除「不在最近 N 条」里的更早记录。
# 内层子查询包一层派生表，绕开 MySQL「不能在子查询里直接 DELETE 同一张表」的限制。
_PRUNE_SQL = """
DELETE FROM user_messages
WHERE username = %s
  AND id NOT IN (
      SELECT keep_id FROM (
          SELECT id AS keep_id FROM user_messages
          WHERE username = %s ORDER BY id DESC LIMIT %s
      ) AS t
  )
"""


class MySQLStore:
    """角色表 + 用户最近对话的 MySQL 读写封装（懒连接，建库建表幂等）。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "rag2",
        charset: str = "utf8mb4",
        recent_n: int = DEFAULT_RECENT_N,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.recent_n = recent_n
        self._db: MySQLClient | None = None

    # ------------------------------------------------------------------ 连接
    def _ensure_database(self) -> None:
        """创建目标数据库（不存在时），需要一条「不选库」的连接。"""
        admin = MySQLClient(
            host=self.host, port=self.port, user=self.user,
            password=self.password, database=None, charset=self.charset,
        )
        try:
            admin.connect()
            admin.execute(
                f"CREATE DATABASE IF NOT EXISTS {MySQLClient._q(self.database)} "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        finally:
            admin.close()

    def client(self) -> MySQLClient:
        """懒加载选中库的连接；首次会先确保数据库存在。"""
        if self._db is None:
            self._ensure_database()
            self._db = MySQLClient(
                host=self.host, port=self.port, user=self.user,
                password=self.password, database=self.database, charset=self.charset,
            )
        return self._db

    def ensure_schema(self) -> None:
        """幂等建表（roles / user_messages）。"""
        db = self.client()
        db.execute(_ROLES_DDL)
        db.execute(_USER_MESSAGES_DDL)
        logger.info("MySQL 表就绪：%s（roles / user_messages）", self.database)

    def ping(self) -> dict[str, Any]:
        db = self.client()
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "version": db.version(),
        }

    # ================================================================== 角色
    @staticmethod
    def _followups_to_json(value: Any) -> str | None:
        return json.dumps(list(value or []), ensure_ascii=False)

    @staticmethod
    def _followups_from_json(raw: Any) -> list[str]:
        if raw is None or raw == "":
            return []
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            return []
        return [str(x) for x in data] if isinstance(data, list) else []

    def load_roles(self) -> list[Role]:
        """从 MySQL 读取全部角色（空表返回 []）。"""
        rows = self.client().query(
            "SELECT role_id, name, system_prompt, description, avatar, followups "
            "FROM roles ORDER BY role_id"
        )
        roles = [
            role_from_dict(
                {
                    "role_id": row["role_id"],
                    "name": row["name"],
                    "system_prompt": row["system_prompt"] or "",
                    "description": row["description"] or "",
                    "avatar": row["avatar"] or "🙂",
                    "followups": self._followups_from_json(row.get("followups")),
                }
            )
            for row in rows
        ]
        return roles

    def save_roles(self, roles: Sequence[Role]) -> None:
        """整表覆盖写入角色（先清空再插入，事务保证一致性）。"""
        rows = [
            {
                "role_id": r.role_id,
                "name": r.name,
                "system_prompt": r.system_prompt,
                "description": r.description or "",
                "avatar": r.avatar or "🙂",
                "followups": self._followups_to_json(r.followups),
            }
            for r in roles
        ]
        db = self.client()
        with db.transaction():
            db.execute("DELETE FROM roles")
            if rows:
                db.insert_many("roles", rows)

    def load_or_seed_roles(self, seed: Sequence[Role]) -> list[Role]:
        """读 MySQL 角色；空表时用 seed 播种一次后返回。"""
        roles = self.load_roles()
        if roles:
            return roles
        if seed:
            self.save_roles(seed)
            return list(seed)
        return []

    # ================================================================== 消息
    def append_messages(self, username: str, messages: Sequence[dict[str, Any]]) -> None:
        """写入一轮消息并裁剪，保证每个用户只留最近 ``recent_n`` 条。"""
        rows = [
            {
                "username": username,
                "role": str(m.get("role", "user")),
                "content": str(m.get("content", "")),
                "role_id": str(m.get("role_id", "")),
                "session_id": str(m.get("session_id", "")),
                "ts": float(m.get("ts", time.time())),
            }
            for m in messages
        ]
        db = self.client()
        with db.transaction():
            db.insert_many("user_messages", rows)
            db.execute(_PRUNE_SQL, (username, username, self.recent_n))

    def recent_messages(self, username: str, limit: int | None = None) -> list[dict[str, Any]]:
        """读某用户最近 ``limit``（默认 recent_n）条消息，按时间正序返回。"""
        n = limit or self.recent_n
        rows = self.client().query(
            "SELECT role, content, role_id, session_id, ts, created_at "
            "FROM user_messages WHERE username = %s ORDER BY id DESC LIMIT %s",
            (username, n),
        )
        return list(reversed(rows))


__all__ = ["MySQLStore", "DEFAULT_RECENT_N"]
