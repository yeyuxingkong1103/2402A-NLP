"""MySQL 存储：角色、用户、知识库文档元数据、会话元数据。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Optional

import pymysql

from app.config import settings
from app.logging_conf import log

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS roles (
        id INT PRIMARY KEY,
        role_name VARCHAR(64) NOT NULL,
        domain VARCHAR(64) NOT NULL DEFAULT 'general',
        description VARCHAR(255) DEFAULT '',
        opening VARCHAR(512) DEFAULT '',
        system_prompt TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(64) PRIMARY KEY,
        nickname VARCHAR(64) DEFAULT '',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS kb_docs (
        doc_id VARCHAR(64) PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        domain VARCHAR(64) NOT NULL DEFAULT 'general',
        parser VARCHAR(32) DEFAULT '',
        chunks INT DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(64) NOT NULL,
        role_id INT NOT NULL,
        turns INT DEFAULT 0,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_session (user_id, role_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


@contextmanager
def get_conn() -> Iterator[pymysql.connections.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_mysql() -> None:
    """建表并写入角色种子数据。"""
    with get_conn() as conn:
        cur = conn.cursor()
        for ddl in _DDL:
            cur.execute(ddl)

        from app.roles.presets import ROLE_PRESETS

        cur.execute("SELECT COUNT(*) AS c FROM roles")
        if cur.fetchone()["c"] == 0:
            for role in ROLE_PRESETS:
                cur.execute(
                    "INSERT INTO roles (id, role_name, domain, description, opening, system_prompt) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        role["id"],
                        role["role_name"],
                        role["domain"],
                        role.get("description", ""),
                        role.get("opening", ""),
                        role["system_prompt"],
                    ),
                )
            log.info("写入 %d 个角色种子", len(ROLE_PRESETS))
    log.info("MySQL 初始化完成")


# ===== 角色 =====
def get_role(role_id: int) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM roles WHERE id=%s", (role_id,))
        return cur.fetchone()


def list_roles() -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, role_name, domain, description, opening FROM roles ORDER BY id")
        return cur.fetchall()


def create_role(role_name: str, domain: str, description: str, opening: str, system_prompt: str) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(id), 0) + 1 AS nid FROM roles")
        nid = cur.fetchone()["nid"]
        cur.execute(
            "INSERT INTO roles (id, role_name, domain, description, opening, system_prompt) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (nid, role_name, domain, description, opening, system_prompt),
        )
        return nid


# ===== 用户 =====
def ensure_user(user_id: str, nickname: str = "") -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (user_id, nickname) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE nickname=VALUES(nickname)",
            (user_id, nickname),
        )


# ===== 知识库文档 =====
def upsert_kb_doc(doc_id: str, filename: str, domain: str, parser: str, chunks: int) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO kb_docs (doc_id, filename, domain, parser, chunks) VALUES (%s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE filename=VALUES(filename), domain=VALUES(domain), "
            "parser=VALUES(parser), chunks=VALUES(chunks)",
            (doc_id, filename, domain, parser, chunks),
        )


def list_kb_docs(domain: Optional[str] = None) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        if domain:
            cur.execute(
                "SELECT * FROM kb_docs WHERE domain=%s ORDER BY updated_at DESC", (domain,)
            )
        else:
            cur.execute("SELECT * FROM kb_docs ORDER BY updated_at DESC")
        rows = cur.fetchall()
    for r in rows:
        for k in ("created_at", "updated_at"):
            if isinstance(r.get(k), datetime):
                r[k] = r[k].strftime("%Y-%m-%d %H:%M:%S")
    return rows


def delete_kb_doc(doc_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM kb_docs WHERE doc_id=%s", (doc_id,))


# ===== 会话 =====
def bump_session(user_id: str, role_id: int) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (user_id, role_id, turns) VALUES (%s, %s, 1) "
            "ON DUPLICATE KEY UPDATE turns=turns+1",
            (user_id, role_id),
        )
