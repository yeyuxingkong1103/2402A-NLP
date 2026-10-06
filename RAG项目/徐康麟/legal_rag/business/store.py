# -*- coding: utf-8 -*-
"""业务数据存储。

表结构对齐设计文档：「MySQL：用户信息、角色信息、会话列表、文档管理表、系统配置表」。
提供两个实现：
  * MySQLBusinessStore —— 真实实现（pymysql），DSN 形如
    mysql://user:password@host:3306/legal_rag
  * SQLiteBusinessStore —— 兜底实现，MySQL 缺席时服务仍可启动

两者共用同一套 DDL 与同一套「先 UPDATE 再 INSERT」的跨库 upsert 写法。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import RagConfig

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id     VARCHAR(64)  PRIMARY KEY,
        name        VARCHAR(128) DEFAULT '',
        created_at  DOUBLE       DEFAULT 0,
        updated_at  DOUBLE       DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roles (
        role_id     VARCHAR(64)  PRIMARY KEY,
        name        VARCHAR(128) DEFAULT '',
        domain      VARCHAR(64)  DEFAULT '',
        persona     TEXT,
        updated_at  DOUBLE       DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id  VARCHAR(128) PRIMARY KEY,
        user_id     VARCHAR(64)  NOT NULL,
        role_id     VARCHAR(64)  NOT NULL,
        title       VARCHAR(255) DEFAULT '',
        created_at  DOUBLE       DEFAULT 0,
        updated_at  DOUBLE       DEFAULT 0,
        title_is_user_set INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id      VARCHAR(128) PRIMARY KEY,
        source      VARCHAR(255) DEFAULT '',
        role_id     VARCHAR(64)  DEFAULT '',
        md5         VARCHAR(64)  DEFAULT '',
        chunks      INTEGER      DEFAULT 0,
        updated_at  DOUBLE       DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        key         VARCHAR(128) PRIMARY KEY,
        value       TEXT,
        updated_at  DOUBLE       DEFAULT 0
    )
    """,
    # B-9（r16 §20）：**会话历史的权威来源**。Redis 只是"近 5 组问答"的上下文窗口，
    # 完整历史一律落这里（含 Redis 裁剪掉的部分），`GET /sessions/{id}/messages`
    # 从这里全量读；`message_id` 做主键 ⇒ 重复投递天然幂等（不会写出第二条）。
    """
    CREATE TABLE IF NOT EXISTS messages (
        message_id  VARCHAR(128) PRIMARY KEY,
        session_id  VARCHAR(128) NOT NULL,
        user_id     VARCHAR(64)  NOT NULL,
        role        VARCHAR(16)  NOT NULL,
        content     TEXT,
        citations   TEXT,
        turn_index  INTEGER      DEFAULT 0,
        seq         INTEGER      DEFAULT 0,
        created_at  DOUBLE       DEFAULT 0,
        updated_at  DOUBLE       DEFAULT 0
    )
    """,
    # 归属过滤（每条 SQL 都带 `WHERE session_id = ? AND user_id = ?`）走这个索引
    """
    CREATE INDEX IF NOT EXISTS idx_messages_owner
        ON messages (session_id, user_id, seq)
    """,
    # B-8（r17 §21）：**账号级偏好**，只有两项 —— `theme` + `role_id`。
    # `theme` 可空：NULL = **未显式选择 = 跟随系统**（必须能与 dark/light 区分，`AC-PR-2`）。
    # 除这两列外**不存任何东西**（不存行为历史 / 设备指纹 / 凭据，`AC-PR-1`）。
    """
    CREATE TABLE IF NOT EXISTS account_prefs (
        user_id     VARCHAR(64)  PRIMARY KEY,
        theme       VARCHAR(16),
        role_id     VARCHAR(64)  DEFAULT '',
        updated_at  DOUBLE       DEFAULT 0
    )
    """,
)

#: 账号体系（注册/登录）专用 DDL。
#:
#: **为什么是独立一张表 `auth_users`，而不是给既有 `users` 加 `password_hash` 列**
#: ------------------------------------------------------------------------------
#: 1. 既有 `users.user_id` 是**任意业务 ID**（`/sessions`、`/chat` 会直接
#:    ``upsert_user("u1")`` 造出没有密码的行）。登录账号必须"**一定有**密码哈希"，
#:    两者语义不同，混在一张表里就会出现"有 user 行但没密码"的半成品，只能靠
#:    ``password_hash IS NOT NULL`` 到处补判空 —— 那是缺陷的温床；
#: 2. 唯一性要**数据库约束兜底**（并发注册不能漏）。既有库（SQLite/MySQL）上
#:    ``ALTER TABLE ... ADD CONSTRAINT UNIQUE`` 不可用（SQLite 不支持），
#:    新表建表时直接写 UNIQUE 才是真正可靠的做法；
#: 3. 建新表对既有 345 项测试**零影响**：`users` 表结构与既往完全一致，
#:    既有路径不读 `auth_users`。
AUTH_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS auth_users (
        user_id       VARCHAR(64)  PRIMARY KEY,
        username      VARCHAR(64)  NOT NULL UNIQUE,
        password_hash TEXT         NOT NULL,
        created_at    DOUBLE       DEFAULT 0,
        updated_at    DOUBLE       DEFAULT 0,
        recovery_salt VARCHAR(64)  DEFAULT '',
        recovery_hash VARCHAR(255) DEFAULT '',
        recovery_rotated_at DOUBLE DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS auth_tokens (
        token_hash  VARCHAR(128) PRIMARY KEY,
        user_id     VARCHAR(64)  NOT NULL,
        issued_at   DOUBLE       DEFAULT 0,
        expires_at  DOUBLE       DEFAULT 0,
        revoked_at  DOUBLE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens (user_id)
    """,
)

#: 建表之后要补的列（**老库升级**用）。
#:
#: 为什么需要它：`CREATE TABLE IF NOT EXISTS` 对**已存在**的表**不会补列** ——
#: 老库（第一批交付时就建好了 `sessions` / `auth_users`）升级到第二批时，
#: 新列必须单独 ALTER 一次。`ALTER TABLE ... ADD COLUMN` 在列已存在时会抛
#: （SQLite/MySQL 都是），因此逐条 try/except：**幂等**、不需要版本表，
#: 也不用引入迁移框架。
COLUMN_MIGRATIONS: tuple[str, ...] = (
    # F-F：标题来源（1 = 用户自己改过，自动标题不得覆盖）
    "ALTER TABLE sessions ADD COLUMN title_is_user_set INTEGER DEFAULT 0",
)

#: 账号表的补列（**只在 ``ensure_auth_schema`` 里跑**）。
#: 分开写的原因：业务表先建时 `auth_users` 可能还不存在，把账号表的 ALTER 混进
#: 业务迁移会刷出一串 "no such table" 告警 —— 明明不是错误，却把日志噪音变成故障信号。
AUTH_COLUMN_MIGRATIONS: tuple[str, ...] = (
    # 一次性恢复码：服务端只存「盐 + 单向哈希」（明文只在注册/重置响应里各出现一次）
    "ALTER TABLE auth_users ADD COLUMN recovery_salt VARCHAR(64) DEFAULT ''",
    "ALTER TABLE auth_users ADD COLUMN recovery_hash VARCHAR(255) DEFAULT ''",
    "ALTER TABLE auth_users ADD COLUMN recovery_rotated_at DOUBLE DEFAULT 0",
)

#: 已知的「唯一约束冲突」异常类型名（跨 sqlite3 / pymysql，不引入新依赖）。
_UNIQUE_VIOLATION_NAMES: frozenset[str] = frozenset({
    "IntegrityError",       # sqlite3.IntegrityError / pymysql.err.IntegrityError
    "OperationalError",     # MySQL 部分版本把 1062 归到这里
    "ProgrammingError",
})


def is_unique_violation(exc: BaseException) -> bool:
    """异常是否是**唯一约束冲突**（而不是别的写入失败）。

    只按异常类型名 + 错误文本判断，不 ``import sqlite3/pymysql`` ——
    业务层不该为了一次判重把两种驱动的异常类型都拖进来。
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if not (names & _UNIQUE_VIOLATION_NAMES):
        return False
    text = str(exc).lower()
    return ("unique" in text) or ("duplicate" in text) or ("1062" in text)


class BusinessStore(ABC):
    name = "base"
    placeholder = "?"

    def __init__(self) -> None:
        self._schema_ready = False
        #: 建表只做一次；多线程同时 ensure_schema 会重复执行 DDL（虽然 IF NOT EXISTS
        #: 幂等，但会放大锁竞争），用锁把它收敛成一次。
        self._schema_lock = threading.Lock()
        #: 账号表（auth_users/auth_tokens）单独的「只建一次」闸门，与业务表的闸门分开：
        #: 不碰账号功能的部署不会因为多一个开关就执行额外 DDL。
        self._auth_ready = False
        self._auth_lock = threading.Lock()

    # ---------- 子类实现 ----------
    @abstractmethod
    def _execute(self, sql: str, params: tuple = ()) -> int:
        """执行写语句，返回受影响行数。**必须线程安全**。"""

    @abstractmethod
    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        """执行读语句，返回字典列表。**必须线程安全**。"""

    @abstractmethod
    def _close(self) -> None:
        ...

    # ---------- 通用 ----------
    def _sql(self, sql: str) -> str:
        return sql if self.placeholder == "?" else sql.replace("?", self.placeholder)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:          # 双重检查：并发下只建一次
                return
            for statement in SCHEMA_STATEMENTS:
                self._execute(statement)
            self._apply_column_migrations()
            self._schema_ready = True
        logger.info("%s 业务表已就绪", self.name)

    def _apply_column_migrations(self, statements: tuple[str, ...] = COLUMN_MIGRATIONS) -> None:
        """老库补列（幂等）：列已存在时 `ALTER TABLE` 会抛，**按预期吞掉**。

        只吞"列已存在"这一种情况，其余异常照抛（否则真出错会被静默吞掉，
        下次读列时才炸，排查成本更高）。
        """
        for statement in statements:
            try:
                self._execute(statement)
            except Exception as exc:  # noqa: BLE001 - 列已存在是预期路径
                text = str(exc).lower()
                if ("duplicate column" in text or "already exists" in text
                        or "duplicate field" in text or "1060" in text):
                    logger.debug("补列跳过（已存在）：%s", statement)
                    continue
                logger.warning("补列失败（将影响该列的读写）：%s -> %s: %s",
                               statement, type(exc).__name__, exc)

    def ensure_auth_schema(self) -> None:
        """建账号体系的两张表（`auth_users` / `auth_tokens`）。

        与 `ensure_schema()` 分开、也各有自己的「只建一次」闸门：
        **没用到账号功能的部署不必碰这两张表**（既有 345 项测试的执行路径
        完全不变），用到时才建，且并发下只建一次。
        """
        if self._auth_ready:
            return
        with self._auth_lock:
            if self._auth_ready:
                return
            for statement in AUTH_SCHEMA_STATEMENTS:
                self._execute(statement)
            self._apply_column_migrations(AUTH_COLUMN_MIGRATIONS)   # 老库补列（恢复码那一组）
            self._auth_ready = True
        logger.info("%s 账号表已就绪（auth_users/auth_tokens）", self.name)

    def _upsert(self, table: str, key_column: str, values: dict) -> None:
        now = time.time()
        values = {**values, "updated_at": now}
        columns = list(values)
        assignments = ", ".join(f"{c} = ?" for c in columns if c != key_column)
        updated = self._execute(
            self._sql(f"UPDATE {table} SET {assignments} WHERE {key_column} = ?"),
            tuple(values[c] for c in columns if c != key_column) + (values[key_column],),
        )
        if updated == 0:
            placeholders = ", ".join("?" for _ in columns)
            self._execute(
                self._sql(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"),
                tuple(values[c] for c in columns),
            )

    # ---------- 用户 ----------
    def upsert_user(self, user_id: str, name: str = "") -> None:
        self.ensure_schema()
        self._upsert("users", "user_id",
                     {"user_id": user_id, "name": name, "created_at": time.time()})

    def get_user(self, user_id: str) -> dict | None:
        self.ensure_schema()
        rows = self._query(self._sql("SELECT * FROM users WHERE user_id = ?"), (user_id,))
        return rows[0] if rows else None

    # ---------- 账号（注册/登录）----------
    def create_auth_user(self, user_id: str, username: str, password_hash: str) -> None:
        """插入一条账号记录；用户名已存在时**由 UNIQUE 约束**抛异常。

        调用方（`legal_rag/api/auth.py`）负责把唯一冲突翻译成 409。
        这里**不做**「先查再插」—— 那种写法在并发注册下会漏（两个请求同时查不到，
        然后都插进去，其中一个被约束挡住才是可靠的）。

        ⚠️ ``password_hash`` 必须是**单向哈希**（PHC 风格串），本层不校验内容、
        不打印它，禁止调用方往里塞明文/可逆密文（见 docs/API.md §9）。
        """
        self.ensure_auth_schema()
        now = time.time()
        self._execute(
            self._sql("INSERT INTO auth_users "
                      "(user_id, username, password_hash, created_at, updated_at) "
                      "VALUES (?, ?, ?, ?, ?)"),
            (user_id, username, password_hash, now, now),
        )

    def get_auth_user(self, user_id: str) -> dict | None:
        self.ensure_auth_schema()
        rows = self._query(
            self._sql("SELECT * FROM auth_users WHERE user_id = ?"), (user_id,))
        return rows[0] if rows else None

    def get_auth_user_by_name(self, username: str) -> dict | None:
        """按**规范化后**的用户名查账号（调用方必须先规范化，本层不猜大小写策略）。"""
        self.ensure_auth_schema()
        rows = self._query(
            self._sql("SELECT * FROM auth_users WHERE username = ?"), (username,))
        return rows[0] if rows else None

    def count_auth_users(self) -> int:
        self.ensure_auth_schema()
        rows = self._query("SELECT COUNT(*) AS n FROM auth_users")
        return int(rows[0]["n"]) if rows else 0

    def create_auth_token(self, token_hash: str, user_id: str,
                          issued_at: float, expires_at: float) -> None:
        """登记一个令牌的**哈希**（绝不存令牌明文）。"""
        self.ensure_auth_schema()
        self._execute(
            self._sql("INSERT INTO auth_tokens "
                      "(token_hash, user_id, issued_at, expires_at, revoked_at) "
                      "VALUES (?, ?, ?, ?, NULL)"),
            (token_hash, user_id, issued_at, expires_at),
        )

    def get_auth_token(self, token_hash: str) -> dict | None:
        self.ensure_auth_schema()
        rows = self._query(
            self._sql("SELECT * FROM auth_tokens WHERE token_hash = ?"), (token_hash,))
        return rows[0] if rows else None

    def revoke_auth_token(self, token_hash: str) -> int:
        """置 ``revoked_at``（**登出后立即失效**：校验侧看的是这个字段，不是删行）。

        返回受影响行数，便于把「登出了一个本来就不存在的令牌」与「真的撤了一个」
        区分开（两者对外都是 204，但日志里能看出差别，排查时有用）。
        """
        self.ensure_auth_schema()
        return self._execute(
            self._sql("UPDATE auth_tokens SET revoked_at = ? "
                      "WHERE token_hash = ? AND revoked_at IS NULL"),
            (time.time(), token_hash),
        )

    def purge_expired_auth_tokens(self, now: float | None = None) -> int:
        """清掉已过期的令牌行（**只清过期**，不动有效令牌；既不拉黑也不误删）。"""
        self.ensure_auth_schema()
        cutoff = time.time() if now is None else now
        return self._execute(
            self._sql("DELETE FROM auth_tokens WHERE expires_at <= ?"), (cutoff,))

    def revoke_all_tokens_for_user(self, user_id: str, *, keep_hash: str = "") -> int:
        """把某用户的**全部有效令牌**置为已撤销（找回密码成功后「旧令牌全部失效」）。

        ``keep_hash`` 非空时保留那一枚（例如"重置后自动登录"用的新令牌）；
        默认不保留 —— 重置的语义就是"此前所有会话都不再可信"。
        """
        self.ensure_auth_schema()
        now = time.time()
        if keep_hash:
            return self._execute(
                self._sql("UPDATE auth_tokens SET revoked_at = ? "
                          "WHERE user_id = ? AND revoked_at IS NULL AND token_hash != ?"),
                (now, user_id, keep_hash))
        return self._execute(
            self._sql("UPDATE auth_tokens SET revoked_at = ? "
                      "WHERE user_id = ? AND revoked_at IS NULL"),
            (now, user_id))

    # ---------- 一次性恢复码（找回密码）----------
    def set_recovery_code(self, user_id: str, salt: str, code_hash: str) -> int:
        """写入/轮换恢复码的「盐 + 单向哈希」（**绝不存明文**）。

        轮换 = 同一行覆盖 + 刷新 ``recovery_rotated_at``；因此**旧码立即作废**
        （校验时拿的是这一行，不保留历史）。
        """
        self.ensure_auth_schema()
        return self._execute(
            self._sql("UPDATE auth_users SET recovery_salt = ?, recovery_hash = ?, "
                      "recovery_rotated_at = ?, updated_at = ? WHERE user_id = ?"),
            (salt, code_hash, time.time(), time.time(), user_id))

    def get_recovery_code(self, user_id: str) -> dict | None:
        """读恢复码记录（盐/哈希/轮换时间）；没有账号或没发过码时返回 ``None``。"""
        self.ensure_auth_schema()
        rows = self._query(
            self._sql("SELECT user_id, recovery_salt, recovery_hash, recovery_rotated_at "
                      "FROM auth_users WHERE user_id = ?"),
            (user_id,))
        if not rows:
            return None
        row = rows[0]
        if not str(row.get("recovery_hash") or ""):
            return None
        return {
            "user_id": str(row["user_id"]),
            "salt": str(row.get("recovery_salt") or ""),
            "hash": str(row.get("recovery_hash") or ""),
            "rotated_at": float(row.get("recovery_rotated_at") or 0.0),
        }

    def update_password_hash(self, user_id: str, password_hash: str) -> int:
        """重置密码：只改哈希（盐在哈希串里自带），并刷新 ``updated_at``。"""
        self.ensure_auth_schema()
        return self._execute(
            self._sql("UPDATE auth_users SET password_hash = ?, updated_at = ? "
                      "WHERE user_id = ?"),
            (password_hash, time.time(), user_id))

    # ---------- 角色 ----------
    def upsert_role(self, role_id: str, name: str, domain: str, persona: str) -> None:
        self.ensure_schema()
        self._upsert("roles", "role_id",
                     {"role_id": role_id, "name": name, "domain": domain, "persona": persona})

    # ---------- 会话 ----------
    def upsert_session(self, session_id: str, user_id: str, role_id: str, title: str = "",
                       *, user_set_title: bool = False) -> int:
        """登记/更新一个会话行；返回**受影响行数**（0 = 该会话属于别人，调用方应 403）。

        标题来源语义（F-F / `AC-TI-*`，用户裁决「用户改名不得被自动标题覆盖」）：

        * ``user_set_title=True``：这是**用户自己**设定的标题（建会话时手填、或 `PATCH`
          改名）→ 写入并把 ``title_is_user_set`` 置 1；
        * ``user_set_title=False``（默认，自动标题 = 首条提问前 30 字）：
          **只在标题为空时写入一次**（`AC-TI-8`：自动标题只写一次）；已经有标题的会话
          **不改写标题**，只推 ``updated_at``（列表按活跃度排序）；
          用户改过名的会话同理（永不被自动标题覆盖）。

        ⚠️ **T75-B1 修复点**：旧实现在 `user_set_title=False` 且非用户设定时**每轮都
        `UPDATE title = 本轮 message[:30]`** ⇒ 第二轮起把首轮落库的标题改短/改乱
        （实测 30 码点 → 15 码点），违反 `AC-TI-8`。现在把「**写标题**」与
        「**只推活跃时间**」两件事在代码里**显式分开**（见下面两个分支与
        :meth:`_touch_session_updated_at`）。

        归属条件（F-G）：行已存在时，``UPDATE`` **必须带 ``user_id`` 条件** ——
        知道别人的 session_id 也不能覆盖对方的标题（返回 0，调用方据此 403），
        这与 `rename_session` / `delete_session` 的口径一致。
        """
        self.ensure_schema()
        existing = self._query(
            self._sql("SELECT session_id, user_id, title, title_is_user_set FROM sessions "
                      "WHERE session_id = ?"),
            (session_id,))
        if existing:
            row = existing[0]
            if str(row.get("user_id") or "") != str(user_id):
                logger.warning("拒绝覆盖他人会话：session_id=%s 属于 %s，请求方 %s",
                               session_id, row.get("user_id"), user_id)
                return 0
            already_user_set = int(row.get("title_is_user_set") or 0)
            if user_set_title:
                return self._execute(
                    self._sql("UPDATE sessions SET title = ?, title_is_user_set = 1, "
                              "updated_at = ? WHERE session_id = ? AND user_id = ?"),
                    (title, time.time(), session_id, user_id))
            if already_user_set:
                # 用户改过名：自动标题**不许**覆盖它，只把 updated_at 往前推（列表按活跃度排序）
                return self._touch_session_updated_at(session_id, user_id)
            if not str(row.get("title") or "").strip():
                # ✅ 自动标题**只写一次**（AC-TI-8）：仅当标题仍为空时写本轮 message[:30]
                return self._execute(
                    self._sql("UPDATE sessions SET title = ?, updated_at = ? "
                              "WHERE session_id = ? AND user_id = ?"),
                    (title, time.time(), session_id, user_id))
            # ✅ 已有自动标题：**不改写标题**，只推 updated_at（列表按活跃度排序）
            return self._touch_session_updated_at(session_id, user_id)
        self._upsert("sessions", "session_id", {
            "session_id": session_id, "user_id": user_id, "role_id": role_id,
            "title": title, "created_at": time.time(),
            "title_is_user_set": 1 if user_set_title else 0,
        })
        return 1

    def _touch_session_updated_at(self, session_id: str, user_id: str) -> int:
        """**只推活跃时间、不碰标题**（会话列表按 `updated_at` 倒序）。

        与"写标题"显式分开：这是**有意的**活跃度信号（用户又来问了一轮），
        不是"无意义地写库"—— 实测见 t95 报告：三轮问答中 `updated_at` 递增、
        `title` 逐字不变。
        """
        return self._execute(
            self._sql("UPDATE sessions SET updated_at = ? "
                      "WHERE session_id = ? AND user_id = ?"),
            (time.time(), session_id, user_id))

    def list_sessions(self, user_id: str) -> list[dict]:
        self.ensure_schema()
        return self._query(
            self._sql("SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC"),
            (user_id,),
        )

    def get_session(self, session_id: str) -> dict | None:
        self.ensure_schema()
        rows = self._query(self._sql("SELECT * FROM sessions WHERE session_id = ?"), (session_id,))
        return rows[0] if rows else None

    def rename_session(self, session_id: str, title: str, user_id: str | None = None) -> int:
        """改会话标题并刷新 ``updated_at``；返回**受影响行数**。

        * 返回 0 表示「这个 session_id 不存在，或者不属于 ``user_id``」——
          调用方据此判 403，而不是假装改成功；
        * 传了 ``user_id`` 时 WHERE 里**带上归属条件**：即使调用方的归属校验
          漏判，也改不到别人的行（不给 TOCTOU 留缝）。
        """
        self.ensure_schema()
        if user_id is None:
            return self._execute(
                self._sql("UPDATE sessions SET title = ?, title_is_user_set = 1, "
                          "updated_at = ? WHERE session_id = ?"),
                (title, time.time(), session_id))
        return self._execute(
            self._sql("UPDATE sessions SET title = ?, title_is_user_set = 1, "
                      "updated_at = ? WHERE session_id = ? AND user_id = ?"),
            (title, time.time(), session_id, user_id))

    def delete_session(self, session_id: str, user_id: str | None = None) -> int:
        """删掉 ``sessions`` 行；返回**受影响行数**（0 = 不存在/不属于该 user）。

        ⚠️ 只删**会话级元数据行**：不动长期记忆（Milvus ``legal_rag_memory``）、
        不动向量库、不动上传的文档；会话历史由调用方用 ``SessionStore.clear``
        按 ``user_id + role_id + session_id`` 清（隔离键必须由服务端补齐）。
        """
        self.ensure_schema()
        if user_id is None:
            return self._execute(self._sql("DELETE FROM sessions WHERE session_id = ?"),
                                 (session_id,))
        return self._execute(
            self._sql("DELETE FROM sessions WHERE session_id = ? AND user_id = ?"),
            (session_id, user_id))

    # ---------- 会话消息（B-9 / r16 §20.3：历史的权威来源） ----------
    def append_messages(self, rows: list[dict]) -> int:
        """把一轮问答的消息写进 `messages` 表；返回**实际新增**行数。

        * **幂等**：``message_id`` 是主键，重复投递（同一 message_id）落库时
          唯一约束冲突按预期吞掉 ⇒ 不产生第二条（`AC-ST-8③`）；
        * 每条 SQL 都**显式带 user_id**（写入值本身来自登录态）—— 归属列非空，
          任何按用户过滤的读都不会串号（`AC-CI-8`）；
        * **半截落库必须可见**（真机实测，2026-09-26）：`message_id` 是**内容寻址**的
          ⇒ 同一会话里重问同一个问题时，**用户消息**因 ID 相同被跳过，而**助手消息**
          的 ID 取决于回答正文（每次不同）⇒ 照写。结果是关系库里留下
          **没有对应提问的"孤儿助手行"**（实测：同会话重问 4 个问题留下 4 条）。
          这里**不改写历史语义**（避免动到既有 AC），只把这件事打成 WARNING + 计数，
          让它在日志与指标里看得见。
        """
        self.ensure_schema()
        inserted = 0
        skipped_roles: set[str] = set()
        inserted_roles: set[str] = set()
        for row in rows or []:
            role = str(row.get("role") or "")
            params = (str(row.get("message_id") or ""),
                      str(row.get("session_id") or ""),
                      str(row.get("user_id") or ""),
                      role,
                      str(row.get("content") or ""),
                      json.dumps(row.get("citations") or [], ensure_ascii=False),
                      int(row.get("turn_index") or 0),
                      int(row.get("seq") or 0),
                      float(row.get("created_at") or 0.0),
                      time.time())
            try:
                self._execute(
                    self._sql("INSERT INTO messages (message_id, session_id, user_id, role, "
                              "content, citations, turn_index, seq, created_at, updated_at) "
                              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"),
                    params)
                inserted += 1
                inserted_roles.add(role)
            except Exception as exc:  # noqa: BLE001 - 唯一冲突 = 重复投递，按预期跳过
                if is_unique_violation(exc):
                    logger.debug("消息已存在，跳过重复写入：message_id=%s", params[0])
                    skipped_roles.add(role)
                    continue
                raise
        if "user" in skipped_roles and "assistant" in inserted_roles:
            session_id = str((rows or [{}])[0].get("session_id") or "")
            logger.warning("同一会话重复问同一个问题：用户消息按幂等被跳过、助手消息却新增 "
                           "⇒ 关系库会出现**没有提问的孤儿助手行**（session=%s）。"
                           "长期记忆侧已按「问题」去重（pair_id），此处的历史语义未改动，"
                           "如需彻底消除请改 message_id 的身份口径（登记在 BACKLOG）",
                           session_id)
            try:
                from .. import metrics as M
                M.counter("orphan_assistant_message_total").inc(session=session_id[:64])
            except Exception:  # noqa: BLE001 - 指标失败不影响主链路
                logger.debug("孤儿助手行计数失败", exc_info=True)
        return inserted

    def list_messages(self, session_id: str, user_id: str) -> list[dict]:
        """按时间序返回该会话的**全量**消息（权威来源，不受 Redis 窗口影响）。"""
        self.ensure_schema()
        return self._query(
            self._sql("SELECT * FROM messages WHERE session_id = ? AND user_id = ? "
                      "ORDER BY seq ASC, created_at ASC"),
            (session_id, user_id))

    def count_messages(self, session_id: str, user_id: str) -> int:
        """该会话的消息条数（**必须带 user_id 条件**，`AC-ST-4` 的对照读数用它）。"""
        self.ensure_schema()
        rows = self._query(
            self._sql("SELECT COUNT(*) AS n FROM messages "
                      "WHERE session_id = ? AND user_id = ?"),
            (session_id, user_id))
        return int((rows[0].get("n") if rows else 0) or 0)

    def count_all_messages(self) -> int:
        """全表条数（只读统计；探针用它做"重复投递后不增长"的前后读数）。"""
        self.ensure_schema()
        rows = self._query(self._sql("SELECT COUNT(*) AS n FROM messages"), ())
        return int((rows[0].get("n") if rows else 0) or 0)

    def next_seq(self, session_id: str, user_id: str) -> int:
        """下一个序号（同一会话内递增；带 user_id 条件，不给他人的会话续号）。"""
        self.ensure_schema()
        rows = self._query(
            self._sql("SELECT MAX(seq) AS m FROM messages "
                      "WHERE session_id = ? AND user_id = ?"),
            (session_id, user_id))
        return int((rows[0].get("m") if rows else 0) or 0) + 1

    def delete_messages(self, session_id: str, user_id: str) -> int:
        """删除该会话的消息（带 user_id 条件）；返回删除行数。"""
        self.ensure_schema()
        return self._execute(
            self._sql("DELETE FROM messages WHERE session_id = ? AND user_id = ?"),
            (session_id, user_id))

    # ---------- 账号级偏好（B-8 / r17 §21：只有 theme + role_id 两项） ----------
    def get_prefs(self, user_id: str) -> dict | None:
        """读该账号的偏好；**没有记录返回 None**（调用方按"未设置"给默认值）。

        每条 SQL 都带 ``WHERE user_id = ?``（`AC-PR-5⑤`）；取值由服务端从登录态派生。
        """
        self.ensure_schema()
        rows = self._query(
            self._sql("SELECT user_id, theme, role_id, updated_at FROM account_prefs "
                      "WHERE user_id = ?"),
            (user_id,))
        if not rows:
            return None
        row = dict(rows[0])
        theme = row.get("theme")
        row["theme"] = None if theme in (None, "", "null") else str(theme)
        row["role_id"] = str(row.get("role_id") or "")
        row["updated_at"] = float(row.get("updated_at") or 0.0)
        return row

    def upsert_prefs(self, user_id: str, **changes) -> dict:
        """写账号偏好（**只认 ``theme`` / ``role_id`` 两个键**）；返回落库后的整行。

        * 未传的键 = 保持原值（部分更新）；
        * ``theme=None`` = **显式**选择"跟随系统"（落库为 NULL），
          与"本次不改主题"**是两件事**（后者=不传这个键）；
        * 额外键一律忽略（字段白名单，`AC-PR-1`）。
        """
        self.ensure_schema()
        allowed = {"theme", "role_id"}
        patch = {key: value for key, value in (changes or {}).items() if key in allowed}
        current = self.get_prefs(user_id)
        theme = current["theme"] if current else None
        role_id = current["role_id"] if current else ""
        if "theme" in patch:
            theme = patch["theme"]
        if "role_id" in patch:
            role_id = patch["role_id"]
        now = time.time()
        updated = self._execute(
            self._sql("UPDATE account_prefs SET theme = ?, role_id = ?, updated_at = ? "
                      "WHERE user_id = ?"),
            (theme, str(role_id or ""), now, user_id))
        if int(updated or 0) == 0:
            self._execute(
                self._sql("INSERT INTO account_prefs (user_id, theme, role_id, updated_at) "
                          "VALUES (?, ?, ?, ?)"),
                (user_id, theme, str(role_id or ""), now))
        return self.get_prefs(user_id) or {
            "user_id": user_id, "theme": theme, "role_id": str(role_id or ""),
            "updated_at": now}

    def delete_prefs(self, user_id: str) -> int:
        """清空该账号的偏好（回到"未设置"：theme=跟随系统、role_id=默认角色）。"""
        self.ensure_schema()
        return self._execute(
            self._sql("DELETE FROM account_prefs WHERE user_id = ?"), (user_id,))

    # ---------- 文档 ----------
    def upsert_document(self, doc_id: str, source: str, role_id: str,
                        md5: str, chunks: int) -> None:
        self.ensure_schema()
        self._upsert("documents", "doc_id", {
            "doc_id": doc_id, "source": source, "role_id": role_id,
            "md5": md5, "chunks": chunks,
        })

    def list_documents(self, role_id: str | None = None) -> list[dict]:
        self.ensure_schema()
        if role_id:
            return self._query(self._sql("SELECT * FROM documents WHERE role_id = ?"), (role_id,))
        return self._query("SELECT * FROM documents")

    # ---------- 配置 ----------
    def set_setting(self, key: str, value: str) -> None:
        self.ensure_schema()
        self._upsert("settings", "key", {"key": key, "value": value})

    def get_setting(self, key: str, default: str = "") -> str:
        self.ensure_schema()
        rows = self._query(self._sql("SELECT value FROM settings WHERE key = ?"), (key,))
        return rows[0]["value"] if rows else default

    # ---------- 运维 ----------
    def health(self) -> dict:
        return {"provider": self.name}

    def close(self) -> None:
        self._close()


class SQLiteBusinessStore(BusinessStore):
    """SQLite 业务库 —— **线程安全**（每线程一个连接 + 写串行）。

    为什么必须改（t4 并发压测抓到的真实缺陷）
    ----------------------------------------
    原实现只用**一个** ``sqlite3.Connection`` 并设 ``check_same_thread=False``，
    服务层把阻塞调用卸载到线程池后，多个 worker 线程并发使用同一连接 ——
    sqlite3 的 C 层状态被并发破坏，表现为
    ``InterfaceError: bad parameter or other API misuse``、
    ``SystemError: error return without exception set``、
    ``IndexError: tuple index out of range``，甚至
    **Windows access violation（弹出模态框把进程卡死）**。
    而"可并发处理用户请求"正是本阶段的核心需求，所以这不是"测试没过"，是**进程会崩**。

    策略（二选一里选了"每线程独立连接 + 写锁"）
    -------------------------------------------
    * **每线程独立连接**（``threading.local``）：这是 sqlite3 官方推荐的安全用法
      （连接不跨线程）。连接数上限 = worker 线程数（默认 ``THREAD_POOL_SIZE=8``），
      每个连接几十 KB，代价可接受；
    * **写操作串行**（``threading.Lock``）：SQLite 是单写者模型，多连接并写会撞
      ``database is locked``；加一把进程内写锁把写排队，比让 sqlite3 反复重试更稳。
      读不加锁（SQLite 支持并发读），因此"读多写少"的会话列表/健康检查不受影响；
    * 附带 ``journal_mode=WAL`` 与 ``busy_timeout``：进一步降低读写互斥与瞬时锁等待。
    """

    name = "sqlite"
    placeholder = "?"
    #: 单个语句等待锁的毫秒数（写锁已串行化，这里只兜底外部进程占用）
    busy_timeout_ms = 5000

    def __init__(self, path) -> None:
        super().__init__()
        import sqlite3

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: list = []          # 已创建的连接（close 时统一关闭）
        self._conn_lock = threading.Lock()    # 保护 _connections 列表
        self._write_lock = threading.Lock()   # 串行化写（SQLite 单写者）

    # ---------- 连接管理 ----------
    def _conn(self):
        """当前线程的连接（不存在就建一个；绝不跨线程复用）。"""
        import sqlite3

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=self.busy_timeout_ms / 1000.0)
            conn.row_factory = sqlite3.Row
            for pragma in (f"PRAGMA busy_timeout={self.busy_timeout_ms}",
                           "PRAGMA journal_mode=WAL",
                           "PRAGMA synchronous=NORMAL"):
                try:
                    conn.execute(pragma)
                except Exception:  # noqa: BLE001 - PRAGMA 失败不影响正确性
                    logger.debug("PRAGMA 失败：%s（忽略）", pragma, exc_info=True)
            self._local.conn = conn
            with self._conn_lock:
                self._connections.append(conn)
            logger.debug("SQLite 新建线程连接：%s（线程=%s，累计 %d）",
                         self.path.name, threading.current_thread().name,
                         len(self._connections))
        return conn

    # ---------- 读写 ----------
    def _execute(self, sql: str, params: tuple = ()) -> int:
        with self._write_lock:                # 写串行：避免 database is locked
            conn = self._conn()
            cursor = conn.execute(sql, params)
            conn.commit()
            return int(cursor.rowcount)

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        # 读不加锁（WAL 下可与写并发）；用本线程自己的连接
        cursor = self._conn().execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def _close(self) -> None:
        with self._conn_lock:
            connections, self._connections = self._connections, []
        self._local = threading.local()
        for conn in connections:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - 关闭失败不阻断退出
                logger.debug("关闭 SQLite 连接失败", exc_info=True)

    def connection_count(self) -> int:
        """已创建的连接数（= 曾用到该库的线程数），便于确认"每线程一连接"。"""
        with self._conn_lock:
            return len(self._connections)


class MySQLBusinessStore(BusinessStore):
    """MySQL 业务库。

    pymysql 的连接**也不是线程安全的**（同一连接并发发查询会串包/报
    ``Packet sequence number wrong``）。因此这里给读写加同一把锁 ——
    业务表是"读多写少"的元数据，串行化的代价可忽略，换来的是并发下不崩。
    （若将来 QPS 上来，应改为连接池，而不是去掉这把锁。）
    """

    name = "mysql"
    placeholder = "%s"

    def __init__(self, dsn: str) -> None:
        super().__init__()
        import pymysql  # type: ignore

        parsed = urllib.parse.urlparse(dsn)
        if parsed.scheme not in ("mysql", "mysql+pymysql"):
            raise ValueError(f"不支持的 MySQL DSN: {dsn}")

        self._lock = threading.Lock()
        self._conn = pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=urllib.parse.unquote(parsed.username or "root"),
            password=urllib.parse.unquote(parsed.password or ""),
            database=(parsed.path or "/").lstrip("/") or None,
            charset="utf8mb4",
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def _execute(self, sql: str, params: tuple = ()) -> int:
        with self._lock, self._conn.cursor() as cursor:
            return cursor.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock, self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall() or [])

    def _close(self) -> None:
        self._conn.close()

    def health(self) -> dict:
        info = dict(super().health())
        try:
            self._query("SELECT 1")
            info["ping"] = True
        except Exception as exc:  # noqa: BLE001
            info["ping"] = False
            info["error"] = str(exc)
        return info


def build_business_store(config: RagConfig) -> BusinessStore:
    """优先 MySQL；连不上就降级到 SQLite 并打印清晰日志。"""
    if config.mysql_dsn:
        try:
            store = MySQLBusinessStore(config.mysql_dsn)
            store.ensure_schema()
            logger.info("业务数据使用 MySQL")
            return store
        except Exception as exc:  # noqa: BLE001 - 降级是预期路径
            logger.warning("MySQL 不可用（%s），业务数据降级为 SQLite", exc)

    path = Path(config.sqlite_path) if config.sqlite_path else (config.index_dir / "legal_rag.db")
    store = SQLiteBusinessStore(path)
    store.ensure_schema()
    logger.info("业务数据使用 SQLite: %s", path)
    return store
