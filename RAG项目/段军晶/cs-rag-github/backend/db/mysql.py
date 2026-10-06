# -*- coding: utf-8 -*-
"""
MySQL 数据访问模块

职责：
    存放并读取文档元信息与 chunk 元数据，是「页码溯源」的唯一真相源。
    本模块只做数据存取，不含任何业务判断。

约定：
    所有函数使用关键字参数，返回 dict 或 dict 列表，便于上层直接序列化。
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pymysql
from pymysql.cursors import DictCursor

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


# ---------------------------------------------------------------------------
# 连接管理
# ---------------------------------------------------------------------------

def _connect(with_database: bool = True, autocommit: bool = False):
    """建立一个 MySQL 连接"""
    kwargs: Dict[str, Any] = {
        "host": settings.mysql_host,
        "port": settings.mysql_port,
        "user": settings.mysql_user,
        "password": settings.mysql_password,
        "charset": settings.mysql_charset,
        "cursorclass": DictCursor,
        "autocommit": autocommit,
    }
    if with_database:
        kwargs["database"] = settings.mysql_database
    return pymysql.connect(**kwargs)


@contextmanager
def get_connection(with_database: bool = True) -> Iterator[Any]:
    """连接上下文管理器：正常结束提交、异常回滚、最终关闭"""
    conn = _connect(with_database=with_database)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

def init_database() -> None:
    """创建业务库（若不存在）。需要账号具备建库权限"""
    try:
        with get_connection(with_database=False) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                    f"DEFAULT CHARSET {settings.mysql_charset} "
                    f"COLLATE {settings.mysql_charset}_unicode_ci"
                )
        logger.info("MySQL 业务库就绪：%s", settings.mysql_database)
    except Exception as exc:
        logger.error("创建 MySQL 业务库失败：%s", exc)
        raise


def init_schema() -> None:
    """执行 schema.sql 建表（幂等，全部为 CREATE TABLE IF NOT EXISTS）"""
    init_database()
    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")

    # 按分号拆分语句，过滤空语句与纯注释
    statements: List[str] = []
    buffer: List[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(buffer).rstrip(";").strip())
            buffer = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                if stmt:
                    cur.execute(stmt)

    logger.info("MySQL 表结构初始化完成（%d 条语句）", len(statements))

    # 已有表不会因 CREATE TABLE IF NOT EXISTS 而新增列，这里做一次幂等补列
    _migrate_schema()


def _migrate_schema() -> None:
    """
    为已存在的表补齐后续版本新增的列（幂等）。

    背景：schema.sql 全部是 CREATE TABLE IF NOT EXISTS，
    对于**已经建好**的库不会改动表结构，因此新增列必须单独做一次判断 + ALTER。
    目前需要补齐的列：users.avatar（V1.1 自选头像）。
    """
    # 需要新增的列
    wanted_columns = {
        "users": {
            "avatar": "ALTER TABLE users ADD COLUMN avatar VARCHAR(32) NOT NULL DEFAULT '' "
                      "COMMENT '头像：emoji 标识或自定义头像地址' AFTER display_name",
        },
    }
    # 需要扩容的列：列名 -> (表名, 需要的最大长度, DDL)
    # 头像从「emoji 标识」扩展成「也可能是 /avatars/xxx.jpg?v=…」后，32 位不够用
    wanted_types = {
        "avatar": (
            "users", 255,
            "ALTER TABLE users MODIFY COLUMN avatar VARCHAR(255) NOT NULL DEFAULT '' "
            "COMMENT '头像：emoji 标识或自定义头像地址（/avatars/xxx.jpg）'",
        ),
    }
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for table, columns in wanted_columns.items():
                    for column, ddl in columns.items():
                        cur.execute(
                            "SELECT COUNT(*) AS n FROM information_schema.COLUMNS "
                            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s",
                            (settings.mysql_database, table, column),
                        )
                        row = cur.fetchone() or {}
                        if int(row.get("n") or 0) == 0:
                            cur.execute(ddl)
                            logger.info("表结构补列完成：%s.%s", table, column)

                for column, (table, need_len, ddl) in wanted_types.items():
                    cur.execute(
                        "SELECT CHARACTER_MAXIMUM_LENGTH AS cur_len "
                        "FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s",
                        (settings.mysql_database, table, column),
                    )
                    row = cur.fetchone() or {}
                    cur_len = row.get("cur_len")
                    if cur_len is not None and int(cur_len) < int(need_len):
                        cur.execute(ddl)
                        logger.info("表结构列扩容完成：%s.%s -> %d", table, column, need_len)
    except Exception as exc:
        logger.warning("表结构补列失败（相关功能可能不可用）：%s", exc)


def health_check() -> bool:
    """连通性检查，供 /api/health 使用"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                cur.fetchone()
        return True
    except Exception as exc:
        logger.error("MySQL 健康检查失败：%s", exc)
        return False


# ---------------------------------------------------------------------------
# 文档元信息
# ---------------------------------------------------------------------------

def upsert_document(
    *,
    doc_id: str,
    file_name: str,
    file_path: str,
    file_hash: str,
    page_count: int = 0,
    chunk_count: int = 0,
    parse_engine: str = "",
    status: str = "pending",
    error_msg: str = "",
) -> None:
    """插入或更新一条文档记录（按 doc_id 覆盖）"""
    sql = """
        INSERT INTO kb_documents
            (doc_id, file_name, file_path, file_hash, page_count,
             chunk_count, parse_engine, status, error_msg)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            file_name    = VALUES(file_name),
            file_path    = VALUES(file_path),
            page_count   = VALUES(page_count),
            chunk_count  = VALUES(chunk_count),
            parse_engine = VALUES(parse_engine),
            status       = VALUES(status),
            error_msg    = VALUES(error_msg)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                doc_id, file_name, file_path, file_hash, page_count,
                chunk_count, parse_engine, status, error_msg,
            ))


def update_document_status(
    *, doc_id: str, status: str, chunk_count: Optional[int] = None,
    page_count: Optional[int] = None, parse_engine: Optional[str] = None,
    error_msg: str = "",
) -> None:
    """更新文档处理状态"""
    sets = ["status = %s", "error_msg = %s"]
    params: List[Any] = [status, error_msg]
    if chunk_count is not None:
        sets.append("chunk_count = %s")
        params.append(chunk_count)
    if page_count is not None:
        sets.append("page_count = %s")
        params.append(page_count)
    if parse_engine is not None:
        sets.append("parse_engine = %s")
        params.append(parse_engine)
    params.append(doc_id)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE kb_documents SET {', '.join(sets)} WHERE doc_id = %s",
                params,
            )


def get_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """按文件哈希查询文档，用于重复上传检测"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM kb_documents WHERE file_hash = %s", (file_hash,))
            return cur.fetchone()


def list_documents() -> List[Dict[str, Any]]:
    """列出全部文档，供知识库状态接口使用"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, file_name, page_count, chunk_count, "
                "       parse_engine, status, created_at "
                "FROM kb_documents ORDER BY created_at"
            )
            return list(cur.fetchall())


def delete_document(doc_id: str) -> None:
    """删除文档及其全部 chunk（外键 ON DELETE CASCADE 自动清理 chunk）"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kb_documents WHERE doc_id = %s", (doc_id,))


# ---------------------------------------------------------------------------
# Chunk 元数据（页码溯源）
# ---------------------------------------------------------------------------

def insert_chunks(chunks: Sequence[Dict[str, Any]]) -> int:
    """
    批量写入 chunk 元数据。

    每个 chunk 必须包含：
        chunk_id, doc_id, chunk_index, page_no, page_nums, content,
        content_type, section_title, source_file
    """
    if not chunks:
        return 0

    sql = """
        INSERT INTO kb_chunks
            (chunk_id, doc_id, chunk_index, page_no, page_nums, content,
             content_type, section_title, source_file, char_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            page_no       = VALUES(page_no),
            page_nums     = VALUES(page_nums),
            content       = VALUES(content),
            content_type  = VALUES(content_type),
            section_title = VALUES(section_title),
            source_file   = VALUES(source_file),
            char_count    = VALUES(char_count)
    """
    rows = [
        (
            c["chunk_id"],
            c["doc_id"],
            int(c.get("chunk_index", 0)),
            int(c.get("page_no", 1)),
            json.dumps(c.get("page_nums", []) or [c.get("page_no", 1)], ensure_ascii=False),
            c.get("content", ""),
            c.get("content_type", "text"),
            c.get("section_title", "") or "",
            c.get("source_file", "") or "",
            len(c.get("content", "") or ""),
        )
        for c in chunks
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)

    logger.debug("写入 %d 条 chunk 元数据", len(rows))
    return len(rows)


def fetch_chunks_by_ids(chunk_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """
    按 chunk_id 批量取回元数据。

    这是检索链路的第 ④ 步：Milvus 只返回「哪些块相关」，
    文件名与页码一律由本函数从 MySQL 取回，保证溯源信息权威可靠。
    """
    if not chunk_ids:
        return []

    placeholders = ",".join(["%s"] * len(chunk_ids))
    sql = f"""
        SELECT chunk_id, doc_id, chunk_index, page_no, page_nums, content,
               content_type, section_title, source_file
        FROM kb_chunks WHERE chunk_id IN ({placeholders})
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, list(chunk_ids))
            rows = list(cur.fetchall())

    # page_nums 由 JSON 字符串还原为列表
    for row in rows:
        raw = row.get("page_nums")
        if isinstance(raw, str):
            try:
                row["page_nums"] = json.loads(raw)
            except (ValueError, TypeError):
                row["page_nums"] = [row.get("page_no", 1)]
        elif raw is None:
            row["page_nums"] = [row.get("page_no", 1)]
    return rows


def fetch_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """按单个 chunk_id 取回元数据"""
    rows = fetch_chunks_by_ids([chunk_id])
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# 统计与问答记录
# ---------------------------------------------------------------------------

def get_kb_stats() -> Dict[str, Any]:
    """汇总知识库规模，供状态接口展示"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS doc_count, "
                "       COALESCE(SUM(page_count), 0)  AS page_count, "
                "       COALESCE(SUM(chunk_count), 0) AS chunk_count "
                "FROM kb_documents WHERE status = 'indexed'"
            )
            stats = cur.fetchone() or {}
            cur.execute("SELECT COUNT(*) AS total_chunks FROM kb_chunks")
            total = cur.fetchone() or {}

    # 注意：MySQL 的 COUNT / SUM 返回 Decimal，直接放进响应体会导致 JSON 序列化失败，
    # 因此在此统一转为 int。
    return {
        "doc_count": int(stats.get("doc_count") or 0),
        "page_count": int(stats.get("page_count") or 0),
        "chunk_count": int(stats.get("chunk_count") or 0),
        "total_chunks": int(total.get("total_chunks") or 0),
    }


def insert_qa_record(
    *,
    session_id: str,
    question: str,
    answer: str,
    sources: Iterable[Dict[str, Any]],
    retrieved: Optional[Iterable[Dict[str, Any]]] = None,
    pipeline: str = "v1",
    latency_ms: int = 0,
) -> None:
    """
    记录一次问答。

    注意：本表仅供后端排查与离线分析，不对外提供查询接口（对齐 N5 约束）。
    """
    sql = """
        INSERT INTO qa_records
            (session_id, question, answer, sources, retrieved, pipeline, latency_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    session_id,
                    question,
                    answer,
                    json.dumps(list(sources), ensure_ascii=False),
                    json.dumps(list(retrieved or []), ensure_ascii=False),
                    pipeline,
                    int(latency_ms),
                ))
    except Exception as exc:
        # 记录失败不应影响问答主流程
        logger.warning("写入问答记录失败（不影响主流程）：%s", exc)


# ---------------------------------------------------------------------------
# 用户账号（增量功能：登录）
# ---------------------------------------------------------------------------
# 课程演示口径：无密码、无加密、无验证码、无 token。
# 账号即身份，输入账号即登录；账号不存在则自动创建。
# 本节的函数只读写 users 表，与知识库/检索/生成链路完全无关。

def get_or_create_user(username: str, avatar: str = "") -> Dict[str, Any]:
    """
    按账号取用户；不存在则创建。返回 {user_id, username, display_name, avatar, created}。

    created 为 True 表示本次是首次登录（新建账号），供接口回给前端提示用。
    并发下同一账号同时首次登录不会产生两条记录：UNIQUE KEY(uk_username) +
    INSERT IGNORE + 回读，保证幂等。

    avatar（V1.1 自选头像）：首次创建时写入；账号已存在时，
    本次传了非空头像且与库中不同则更新 —— 即「同一个账号可以换头像」。
    """
    name = (username or "").strip()
    new_avatar = (avatar or "").strip()[:16]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, username, display_name, avatar, created_at FROM users "
                "WHERE username = %s",
                (name,),
            )
            row = cur.fetchone()
            created = False
            if not row:
                cur.execute(
                    "INSERT IGNORE INTO users "
                    "(user_id, username, display_name, avatar, last_login_at) "
                    "VALUES (REPLACE(UUID(), '-', ''), %s, %s, %s, NOW())",
                    (name, name, new_avatar),
                )
                cur.execute(
                    "SELECT user_id, username, display_name, avatar, created_at FROM users "
                    "WHERE username = %s",
                    (name,),
                )
                row = cur.fetchone()
                created = True
            if row:
                if new_avatar and new_avatar != (row.get("avatar") or ""):
                    cur.execute(
                        "UPDATE users SET avatar = %s WHERE user_id = %s",
                        (new_avatar, row["user_id"]),
                    )
                    row["avatar"] = new_avatar
                cur.execute(
                    "UPDATE users SET last_login_at = NOW() WHERE user_id = %s",
                    (row["user_id"],),
                )
    if not row:
        raise RuntimeError("账号创建失败，请稍后重试")
    if not row.get("display_name"):
        row["display_name"] = name
    if row.get("avatar") is None:
        row["avatar"] = ""
    row["created"] = created
    return row


def update_user_avatar(user_id: str, avatar: str) -> None:
    """更新用户头像（emoji 标识，或自定义头像地址 /avatars/xxx.jpg）"""
    if not user_id:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET avatar = %s WHERE user_id = %s",
                ((avatar or "")[:255], user_id),
            )


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """按 user_id 取用户，用于校验前端传来的 user_id 是否真实存在"""
    if not user_id:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, username, display_name, avatar, created_at FROM users "
                "WHERE user_id = %s",
                (user_id,),
            )
            return cur.fetchone()


# ---------------------------------------------------------------------------
# 对话会话与消息（增量功能：历史记录）
# ---------------------------------------------------------------------------
# 分工说明：
#   Redis  —— 短生命周期多轮上下文（SESSION_TTL_SECONDS 到期即失效）
#   MySQL  —— 长期历史记录（不清除就永远在，清除浏览器缓存不受影响）
# 两者共用同一个 session_id，互不干扰。

def get_session_owner(session_id: str) -> Optional[str]:
    """取会话归属的 user_id；会话不存在返回 None"""
    if not session_id:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    return row["user_id"] if row else None


def upsert_chat_session(
    *,
    session_id: str,
    user_id: str,
    role: str = "student",
    title: str = "",
) -> None:
    """
    建立或更新会话记录。
    title 只在首次写入时生效（会话标题 = 首个提问的前 12 字），
    后续提问不会覆盖已有标题；role 与 updated_at 每次刷新。
    message_count 由自增语句维护。
    """
    if not session_id or not user_id:
        return
    sql = """
        INSERT INTO chat_sessions (session_id, user_id, title, role, message_count)
        VALUES (%s, %s, %s, %s, 0)
        ON DUPLICATE KEY UPDATE
            role          = VALUES(role),
            updated_at    = CURRENT_TIMESTAMP,
            title         = IF(title = '', VALUES(title), title)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, user_id, (title or "")[:12], role))


def insert_chat_message(
    *,
    session_id: str,
    user_id: str,
    question: str,
    answer: str,
    sources: Iterable[Dict[str, Any]],
    role: str = "student",
    latency_ms: int = 0,
) -> int:
    """写入一条问答消息，返回自增 id（前端收藏用的 message_id）"""
    sql = """
        INSERT INTO chat_messages
            (session_id, user_id, question, answer, sources, role, latency_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                session_id,
                user_id,
                question,
                answer,
                json.dumps(list(sources), ensure_ascii=False),
                role,
                int(latency_ms),
            ))
            message_id = int(cur.lastrowid or 0)
            cur.execute(
                "UPDATE chat_sessions SET message_count = message_count + 1, "
                "updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
                (session_id,),
            )
    return message_id


def _decode_sources(row: Dict[str, Any]) -> Dict[str, Any]:
    """把 sources 列的 JSON 字符串还原为列表（pymysql 对 JSON 列返回字符串）"""
    raw = row.get("sources")
    if isinstance(raw, str):
        try:
            row["sources"] = json.loads(raw)
        except (ValueError, TypeError):
            row["sources"] = []
    elif raw is None:
        row["sources"] = []
    return row


def list_user_sessions(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """列出某用户的全部会话（历史记录左侧列表），按最近问答时间倒序"""
    if not user_id:
        return []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, title, role, message_count, created_at, updated_at "
                "FROM chat_sessions WHERE user_id = %s "
                "ORDER BY updated_at DESC, created_at DESC LIMIT %s",
                (user_id, int(limit)),
            )
            return list(cur.fetchall())


def get_session_messages(
    *, session_id: str, user_id: str, limit: int = 200
) -> List[Dict[str, Any]]:
    """取某会话下的全部问答（按时间正序），用于打开历史会话时回显"""
    if not session_id or not user_id:
        return []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, session_id, question, answer, sources, role, latency_ms, created_at "
                "FROM chat_messages WHERE session_id = %s AND user_id = %s "
                "ORDER BY id ASC LIMIT %s",
                (session_id, user_id, int(limit)),
            )
            rows = list(cur.fetchall())
    return [_decode_sources(r) for r in rows]


def delete_chat_session(*, session_id: str, user_id: str) -> int:
    """
    删除会话及其全部消息（外键 ON DELETE CASCADE 自动清理 chat_messages）。
    只允许删除自己的会话：SQL 里带 user_id 条件。
    收藏表不加外键，删除会话不会连带删掉用户已收藏的快照。
    """
    if not session_id or not user_id:
        return 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_sessions WHERE session_id = %s AND user_id = %s",
                (session_id, user_id),
            )
            return int(cur.rowcount or 0)


def delete_chat_message(*, message_id: int, user_id: str) -> int:
    """删除单条问答（只允许删自己的）"""
    if not message_id or not user_id:
        return 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 先取所属会话（删除后就查不到了），再删除并维护会话内计数
            cur.execute(
                "SELECT session_id FROM chat_messages WHERE id = %s AND user_id = %s",
                (int(message_id), user_id),
            )
            _row = cur.fetchone()
            if not _row:
                return 0
            _session_id = _row["session_id"]
            cur.execute(
                "DELETE FROM chat_messages WHERE id = %s AND user_id = %s",
                (int(message_id), user_id),
            )
            deleted = int(cur.rowcount or 0)
            if deleted:
                # 维护会话内的消息计数，避免负数
                cur.execute(
                    "UPDATE chat_sessions SET message_count = GREATEST(message_count - 1, 0) "
                    "WHERE session_id = %s",
                    (_session_id,),
                )
    return deleted


def get_chat_message(*, message_id: int, user_id: str) -> Optional[Dict[str, Any]]:
    """取单条问答（含溯源快照），收藏时以服务端数据为准，防止前端伪造内容"""
    if not message_id or not user_id:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, session_id, question, answer, sources, role, created_at "
                "FROM chat_messages WHERE id = %s AND user_id = %s",
                (int(message_id), user_id),
            )
            row = cur.fetchone()
    return _decode_sources(row) if row else None


# ---------------------------------------------------------------------------
# 用户收藏（增量功能）
# ---------------------------------------------------------------------------
# 收藏存整条快照（问题 + 答案 + 溯源），知识库重建后收藏仍可回看。

def add_favorite(
    *,
    user_id: str,
    message_id: Optional[int],
    session_id: str = "",
    question: str,
    answer: str,
    sources: Iterable[Dict[str, Any]],
    role: str = "student",
) -> Dict[str, Any]:
    """
    收藏一条问答。同一用户对同一条消息重复收藏不报错（幂等），
    返回 {id, message_id, already}。
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if message_id:
                cur.execute(
                    "SELECT id FROM user_favorites WHERE user_id = %s AND message_id = %s",
                    (user_id, int(message_id)),
                )
                existing = cur.fetchone()
                if existing:
                    return {"id": int(existing["id"]), "message_id": int(message_id), "already": True}

            cur.execute(
                "INSERT INTO user_favorites "
                "(user_id, message_id, session_id, question, answer, sources, role) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    user_id,
                    int(message_id) if message_id else None,
                    session_id or "",
                    question,
                    answer,
                    json.dumps(list(sources), ensure_ascii=False),
                    role,
                ),
            )
            return {"id": int(cur.lastrowid or 0), "message_id": message_id, "already": False}


def remove_favorite(*, user_id: str, message_id: Optional[int] = None,
                    favorite_id: Optional[int] = None) -> int:
    """取消收藏：优先按 favorite_id 删，其次按 message_id 删（都带 user_id 约束）"""
    if not user_id:
        return 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            if favorite_id:
                cur.execute(
                    "DELETE FROM user_favorites WHERE id = %s AND user_id = %s",
                    (int(favorite_id), user_id),
                )
            elif message_id:
                cur.execute(
                    "DELETE FROM user_favorites WHERE user_id = %s AND message_id = %s",
                    (user_id, int(message_id)),
                )
            else:
                return 0
            return int(cur.rowcount or 0)


def list_favorites(user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """列出某用户的全部收藏，按收藏时间倒序"""
    if not user_id:
        return []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, message_id, session_id, question, answer, sources, "
                "       role, created_at "
                "FROM user_favorites WHERE user_id = %s "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                (user_id, int(limit)),
            )
            rows = list(cur.fetchall())
    return [_decode_sources(r) for r in rows]