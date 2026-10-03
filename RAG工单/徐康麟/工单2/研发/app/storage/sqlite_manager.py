"""SQLite 存储层：文档、分块、对话、消息、反馈、评估、检索追踪与日志索引。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 存储层（对应 设计/接口设计.md §2.16、§6）

设计要点：
- 建表用原生 SQL（``_SCHEMA``），复杂查询亦用原生 SQL，便于审计。
- 连接级 ``check_same_thread=False`` + 每次操作独立连接，支持 Streamlit 多线程。
- 所有写操作自带事务与异常日志，禁止静默失败。
- 基线 8 张表沿用（documents / chunks / conversations / messages / feedback /
  eval_results / golden_qa / logs），本工单**新增 retrieval_traces**（检索片段全记录），
  并给 chunks 增加 is_boilerplate 列（释义页标记，幂等迁移）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.errors import StorageError
from app.core.logging_conf import logger, trace
from app.models.schemas import (
    Chunk,
    Citation,
    Conversation,
    DocumentMeta,
    EvalRecord,
    Feedback,
    GoldenQA,
    Message,
    RetrievalTrace,
)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- 文档元数据
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    source_path  TEXT NOT NULL DEFAULT '',
    page_count   INTEGER NOT NULL DEFAULT 0,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    table_count  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'pending',
    is_default   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

-- 分块内容与元数据
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    page        INTEGER NOT NULL,
    section     TEXT NOT NULL DEFAULT '',
    type        TEXT NOT NULL DEFAULT 'text',
    content     TEXT NOT NULL,
    char_count  INTEGER NOT NULL DEFAULT 0,
    table_id    TEXT,
    keywords    TEXT NOT NULL DEFAULT '[]',
    is_boilerplate INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc   ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page  ON chunks(doc_id, page);
CREATE INDEX IF NOT EXISTS idx_chunks_type  ON chunks(doc_id, type);
CREATE INDEX IF NOT EXISTS idx_chunks_table ON chunks(table_id);

-- 对话会话
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '新对话',
    doc_id          TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    message_count   INTEGER NOT NULL DEFAULT 0
);

-- 消息记录
CREATE TABLE IF NOT EXISTS messages (
    message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations       TEXT NOT NULL DEFAULT '[]',
    first_token_ms  REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, message_id);

-- 用户反馈
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    message_id      INTEGER,
    rating          TEXT NOT NULL,
    comment         TEXT NOT NULL DEFAULT '',
    question        TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

-- 评估结果
CREATE TABLE IF NOT EXISTS eval_results (
    eval_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id        INTEGER NOT NULL,
    question           TEXT NOT NULL,
    mode               TEXT NOT NULL,
    answer             TEXT NOT NULL,
    golden             TEXT NOT NULL DEFAULT '',
    is_correct         INTEGER NOT NULL DEFAULT 0,
    is_unknown         INTEGER NOT NULL DEFAULT 0,
    should_be_unknown  INTEGER NOT NULL DEFAULT 0,
    citation_pages     TEXT NOT NULL DEFAULT '[]',
    citation_valid     INTEGER NOT NULL DEFAULT 0,
    first_token_ms     REAL NOT NULL DEFAULT 0,
    total_ms           REAL NOT NULL DEFAULT 0,
    faithfulness       REAL,
    answer_relevancy   REAL,
    context_precision  REAL,
    context_recall     REAL,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_q ON eval_results(question_id, mode);

-- 标准问答（golden_qa.jsonl 的数据库镜像，便于 SQL 校验）
CREATE TABLE IF NOT EXISTS golden_qa (
    id            INTEGER PRIMARY KEY,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    evidence      TEXT NOT NULL DEFAULT '',
    evidence_pages TEXT NOT NULL DEFAULT '[]',
    category      TEXT NOT NULL DEFAULT '',
    should_be_unknown INTEGER NOT NULL DEFAULT 0
);

-- 关键日志索引
CREATE TABLE IF NOT EXISTS logs (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    level      TEXT NOT NULL DEFAULT 'INFO',
    module     TEXT NOT NULL DEFAULT '',
    function   TEXT NOT NULL DEFAULT '',
    message    TEXT NOT NULL DEFAULT '',
    payload    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_logs_ts     ON logs(ts);
CREATE INDEX IF NOT EXISTS idx_logs_level  ON logs(level);

-- 【本工单新增】检索追踪：一次提问的候选块 / 变体 / 加权 / 页码 / 耗时全记录
CREATE TABLE IF NOT EXISTS retrieval_traces (
    trace_id        TEXT PRIMARY KEY,          -- 一次提问的贯穿 ID（与日志 trace_id 相同）
    conversation_id TEXT,                      -- 可空（离线评估无会话）
    message_id      INTEGER,                   -- 关联 messages.message_id（生成后可回填）
    question        TEXT NOT NULL,             -- 用户原始问题
    rewritten_query TEXT NOT NULL DEFAULT '',  -- 多轮改写后的独立问题
    variants        TEXT NOT NULL DEFAULT '[]',-- 查询变体（JSON 数组）
    candidates      TEXT NOT NULL DEFAULT '[]',-- 候选块（JSON：chunk_id/page/type/score/vector_score/bm25_score/rerank_score）
    boosts          TEXT NOT NULL DEFAULT '{}',-- 各加权/降权系数汇总（JSON）
    final_pages     TEXT NOT NULL DEFAULT '[]',-- 送入生成的页码（JSON）
    rerank_mode     TEXT NOT NULL DEFAULT 'off', -- model | rule | off
    top_cosine      REAL NOT NULL DEFAULT 0,   -- 最高原始余弦（拒答判据）
    first_token_ms  REAL NOT NULL DEFAULT 0,
    total_ms        REAL NOT NULL DEFAULT 0,
    mode            TEXT NOT NULL DEFAULT 'llm', -- llm | extractive | fallback
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_conv ON retrieval_traces(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_traces_q    ON retrieval_traces(question);
"""


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc).astimezone()).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _unjson(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class SQLiteManager:
    """SQLite 管理器：建表 + 各实体的增删改查。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else settings.paths.sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init_schema()

    # ------------------------------------------------------------------
    # 连接与建表
    # ------------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        """新建一个连接（调用方负责关闭，推荐用 ``session()``）。

        注意：``PRAGMA foreign_keys`` 是**连接级**开关，``executescript`` 里写一次
        只对建表那条连接生效。若不在这里逐连接开启，外键约束实际上是不生效的
        （实测 ``PRAGMA foreign_keys`` 返回 0，孤儿分块可写入、ON DELETE CASCADE 失效）。
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        """事务上下文：正常提交，异常回滚并记录堆栈后抛出。"""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("app.storage.sqlite_manager", "SQLite 事务失败，已回滚", db=str(self.db_path))
            raise
        finally:
            conn.close()

    @trace
    def init_schema(self) -> None:
        """初始化全部表结构（幂等，可重复执行），并执行列级迁移。"""
        with self._lock, self.session() as conn:
            conn.executescript(_SCHEMA)
            # 列级迁移：基线库可能已存在而无 is_boilerplate 列（幂等）
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "is_boilerplate" not in columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN is_boilerplate INTEGER NOT NULL DEFAULT 0")
                logger.info("app.storage.sqlite_manager", "已为 chunks 表补充 is_boilerplate 列（迁移）")

    def table_names(self) -> list[str]:
        """返回当前数据库的全部表名（测试用）。"""
        with self.session() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return [row["name"] for row in rows]

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------
    @trace
    def upsert_document(self, meta: DocumentMeta) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, title, source_path, page_count, chunk_count,
                                       table_count, status, is_default, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title, source_path=excluded.source_path,
                    page_count=excluded.page_count, chunk_count=excluded.chunk_count,
                    table_count=excluded.table_count, status=excluded.status,
                    is_default=excluded.is_default, updated_at=excluded.updated_at
                """,
                (
                    meta.doc_id,
                    meta.title,
                    meta.source_path,
                    meta.page_count,
                    meta.chunk_count,
                    meta.table_count,
                    meta.status,
                    int(meta.is_default),
                    _iso(meta.created_at),
                    _iso(meta.updated_at),
                ),
            )

    def list_documents(self) -> list[DocumentMeta]:
        with self.session() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY is_default DESC, created_at DESC").fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, doc_id: str) -> DocumentMeta | None:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def get_default_document(self) -> DocumentMeta | None:
        with self.session() as conn:
            row = conn.execute(
                "SELECT * FROM documents ORDER BY is_default DESC, created_at DESC LIMIT 1"
            ).fetchone()
        return self._row_to_document(row) if row else None

    @trace
    def delete_document(self, doc_id: str) -> int:
        """删除文档及其全部分块，返回删除的分块数。"""
        with self.session() as conn:
            removed = conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,)).rowcount
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        logger.info("app.storage.sqlite_manager", "删除文档", doc_id=doc_id, removed_chunks=removed)
        return removed

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentMeta:
        return DocumentMeta(
            doc_id=row["doc_id"],
            title=row["title"],
            source_path=row["source_path"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            table_count=row["table_count"],
            status=row["status"],
            is_default=bool(row["is_default"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ------------------------------------------------------------------
    # chunks
    # ------------------------------------------------------------------
    @trace
    def insert_chunks(self, chunks: Iterable[Chunk], replace_doc: bool = True) -> int:
        """批量写入分块；``replace_doc=True`` 时先清空该文档旧分块。"""
        items = list(chunks)
        if not items:
            return 0
        doc_id = items[0].doc_id
        with self.session() as conn:
            if replace_doc:
                conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, doc_id, page, section, type, content,
                                    char_count, table_id, keywords, is_boilerplate)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    page=excluded.page, section=excluded.section, type=excluded.type,
                    content=excluded.content, char_count=excluded.char_count,
                    table_id=excluded.table_id, keywords=excluded.keywords,
                    is_boilerplate=excluded.is_boilerplate
                """,
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.page,
                        c.section,
                        c.type,
                        c.content,
                        c.char_count or len(c.content),
                        c.table_id,
                        _json(c.keywords),
                        1 if c.is_boilerplate else 0,
                    )
                    for c in items
                ],
            )
        logger.info("app.storage.sqlite_manager", "写入分块", doc_id=doc_id, count=len(items))
        return len(items)

    def get_chunks(self, doc_id: str, chunk_type: str | None = None) -> list[Chunk]:
        sql = "SELECT * FROM chunks WHERE doc_id=?"
        params: list[Any] = [doc_id]
        if chunk_type:
            sql += " AND type=?"
            params.append(chunk_type)
        sql += " ORDER BY page, chunk_id"
        with self.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self.session() as conn:
            rows = conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids).fetchall()
        by_id = {row["chunk_id"]: self._row_to_chunk(row) for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def count_chunks(self, doc_id: str | None = None) -> int:
        with self.session() as conn:
            if doc_id:
                row = conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def page_range(self, doc_id: str) -> tuple[int, int]:
        """返回该文档分块覆盖的页码区间（引用校验用）。"""
        with self.session() as conn:
            row = conn.execute(
                "SELECT MIN(page) AS lo, MAX(page) AS hi FROM chunks WHERE doc_id=?", (doc_id,)
            ).fetchone()
        return (int(row["lo"] or 0), int(row["hi"] or 0))

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            page=row["page"],
            section=row["section"],
            type=row["type"],
            content=row["content"],
            char_count=row["char_count"],
            table_id=row["table_id"],
            keywords=_unjson(row["keywords"], []),
            is_boilerplate=bool(row["is_boilerplate"]) if "is_boilerplate" in row.keys() else False,
        )

    # ------------------------------------------------------------------
    # conversations / messages
    # ------------------------------------------------------------------
    @trace
    def create_conversation(self, conversation_id: str, title: str = "新对话", doc_id: str | None = None) -> Conversation:
        """创建会话；已存在时**只更新标题与文档**，不重置统计字段。

        原先用 ``INSERT OR REPLACE`` 会把已存在的行整行替换掉，
        ``message_count`` 被重置为 0、``created_at`` 被覆盖，
        导致界面显示的会话消息数永远为 0，
        并且 ``auto_title`` 里 ``message_count > 2`` 的守卫永不成立，
        每一轮追问都会把标题改成最后一个问题。
        """
        now = _iso()
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title, doc_id, created_at, updated_at, message_count)
                VALUES (?,?,?,?,?,0)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    title=excluded.title,
                    doc_id=COALESCE(excluded.doc_id, conversations.doc_id),
                    updated_at=excluded.updated_at
                """,
                (conversation_id, title, doc_id, now, now),
            )
        return Conversation(conversation_id=conversation_id, title=title, doc_id=doc_id)

    @trace
    def set_conversation_title(self, conversation_id: str, title: str) -> None:
        """只更新标题（供首轮问题自动命名使用，不动其它字段）。"""
        with self.session() as conn:
            conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE conversation_id=?",
                (title, _iso(), conversation_id),
            )

    def list_conversations(self) -> list[Conversation]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [
            Conversation(
                conversation_id=row["conversation_id"],
                title=row["title"],
                doc_id=row["doc_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self.session() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
        if not row:
            return None
        return Conversation(
            conversation_id=row["conversation_id"],
            title=row["title"],
            doc_id=row["doc_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            message_count=row["message_count"],
        )

    @trace
    def add_message(self, message: Message) -> int:
        with self.session() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, citations, first_token_ms, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    message.conversation_id,
                    message.role,
                    message.content,
                    _json([c.model_dump() for c in message.citations]),
                    message.first_token_ms,
                    _iso(message.created_at),
                ),
            )
            message_id = int(cursor.lastrowid or 0)
            conn.execute(
                "UPDATE conversations SET updated_at=?, message_count=message_count+1 WHERE conversation_id=?",
                (_iso(), message.conversation_id),
            )
        return message_id

    def get_messages(self, conversation_id: str, limit: int | None = None) -> list[Message]:
        sql = "SELECT * FROM messages WHERE conversation_id=? ORDER BY message_id"
        params: list[Any] = [conversation_id]
        if limit:
            sql = (
                "SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY message_id DESC LIMIT ?)"
                " ORDER BY message_id"
            )
            params.append(limit)
        with self.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            Message(
                message_id=row["message_id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                citations=[Citation(**item) for item in _unjson(row["citations"], [])],
                first_token_ms=row["first_token_ms"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    @trace
    def clear_conversation(self, conversation_id: str) -> int:
        with self.session() as conn:
            removed = conn.execute(
                "DELETE FROM messages WHERE conversation_id=?", (conversation_id,)
            ).rowcount
            conn.execute(
                "UPDATE conversations SET message_count=0, updated_at=? WHERE conversation_id=?",
                (_iso(), conversation_id),
            )
        return removed

    @trace
    def delete_conversation(self, conversation_id: str) -> None:
        with self.session() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))

    # ------------------------------------------------------------------
    # feedback
    # ------------------------------------------------------------------
    @trace
    def add_feedback(self, feedback: Feedback) -> int:
        with self.session() as conn:
            cursor = conn.execute(
                "INSERT INTO feedback (conversation_id, message_id, rating, comment, question, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    feedback.conversation_id,
                    feedback.message_id,
                    feedback.rating,
                    feedback.comment,
                    feedback.question,
                    _iso(feedback.created_at),
                ),
            )
        return int(cursor.lastrowid or 0)

    def list_feedback(self, limit: int = 100) -> list[Feedback]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY feedback_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            Feedback(
                feedback_id=row["feedback_id"],
                conversation_id=row["conversation_id"],
                message_id=row["message_id"],
                rating=row["rating"],
                comment=row["comment"],
                question=row["question"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def feedback_stats(self) -> dict[str, int]:
        with self.session() as conn:
            rows = conn.execute("SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating").fetchall()
        stats = {"up": 0, "down": 0}
        for row in rows:
            stats[row["rating"]] = int(row["n"])
        return stats

    # ------------------------------------------------------------------
    # eval_results
    # ------------------------------------------------------------------
    @trace
    def add_eval_record(self, record: EvalRecord) -> int:
        with self.session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO eval_results (question_id, question, mode, answer, golden, is_correct,
                    is_unknown, should_be_unknown, citation_pages, citation_valid, first_token_ms,
                    total_ms, faithfulness, answer_relevancy, context_precision, context_recall, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.question_id,
                    record.question,
                    record.mode,
                    record.answer,
                    record.golden,
                    int(record.is_correct),
                    int(record.is_unknown),
                    int(record.should_be_unknown),
                    _json(record.citation_pages),
                    int(record.citation_valid_count),
                    record.first_token_ms,
                    record.total_ms,
                    record.faithfulness,
                    record.answer_relevancy,
                    record.context_precision,
                    record.context_recall,
                    _iso(record.created_at),
                ),
            )
        return int(cursor.lastrowid or 0)

    def list_eval_records(self, mode: str | None = None) -> list[EvalRecord]:
        sql = "SELECT * FROM eval_results"
        params: list[Any] = []
        if mode:
            sql += " WHERE mode=?"
            params.append(mode)
        sql += " ORDER BY question_id, eval_id"
        with self.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            EvalRecord(
                eval_id=row["eval_id"],
                question_id=row["question_id"],
                question=row["question"],
                mode=row["mode"],
                answer=row["answer"],
                golden=row["golden"],
                is_correct=bool(row["is_correct"]),
                is_unknown=bool(row["is_unknown"]),
                should_be_unknown=bool(row["should_be_unknown"]),
                citation_pages=_unjson(row["citation_pages"], []),
                # 该列存的是“有效引用条数”（整数），保留 bool 字段以兼容旧读取方
                citation_valid=bool(row["citation_valid"]),
                citation_valid_count=int(row["citation_valid"] or 0),
                first_token_ms=row["first_token_ms"],
                total_ms=row["total_ms"],
                faithfulness=row["faithfulness"],
                answer_relevancy=row["answer_relevancy"],
                context_precision=row["context_precision"],
                context_recall=row["context_recall"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    @trace
    def clear_eval_records(self) -> None:
        with self.session() as conn:
            conn.execute("DELETE FROM eval_results")

    # ------------------------------------------------------------------
    # golden_qa
    # ------------------------------------------------------------------
    @trace
    def upsert_golden_qa(self, items: Iterable[GoldenQA]) -> int:
        rows = list(items)
        with self.session() as conn:
            conn.executemany(
                """
                INSERT INTO golden_qa (id, question, answer, evidence, evidence_pages, category, should_be_unknown)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    question=excluded.question, answer=excluded.answer, evidence=excluded.evidence,
                    evidence_pages=excluded.evidence_pages, category=excluded.category,
                    should_be_unknown=excluded.should_be_unknown
                """,
                [
                    (g.id, g.question, g.answer, g.evidence, _json(g.evidence_pages), g.category, int(g.should_be_unknown))
                    for g in rows
                ],
            )
        return len(rows)

    def list_golden_qa(self) -> list[GoldenQA]:
        with self.session() as conn:
            rows = conn.execute("SELECT * FROM golden_qa ORDER BY id").fetchall()
        return [
            GoldenQA(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                evidence=row["evidence"],
                evidence_pages=_unjson(row["evidence_pages"], []),
                category=row["category"],
                should_be_unknown=bool(row["should_be_unknown"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # logs（关键日志索引）
    # ------------------------------------------------------------------
    @trace
    def add_log(self, module: str, message: str, level: str = "INFO", function: str = "", payload: dict[str, Any] | None = None) -> None:
        with self.session() as conn:
            conn.execute(
                "INSERT INTO logs (ts, level, module, function, message, payload) VALUES (?,?,?,?,?,?)",
                (_iso(), level, module, function, message, _json(payload or {})),
            )

    def list_logs(self, level: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM logs"
        params: list[Any] = []
        if level:
            sql += " WHERE level=?"
            params.append(level)
        sql += " ORDER BY log_id DESC LIMIT ?"
        params.append(limit)
        with self.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        """返回各表行数，供界面与健康检查使用。"""
        tables = [
            "documents",
            "chunks",
            "conversations",
            "messages",
            "feedback",
            "eval_results",
            "golden_qa",
            "logs",
            "retrieval_traces",
        ]
        out: dict[str, int] = {}
        with self.session() as conn:
            for table in tables:
                out[table] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
        return out

    # ------------------------------------------------------------------
    # 【本工单新增】检索追踪 retrieval_traces
    # ------------------------------------------------------------------
    @trace
    def add_retrieval_trace(self, trace: RetrievalTrace) -> int:
        """写入一条检索追踪（同 trace_id 重复写入时覆盖，保证一次提问一行）。"""
        try:
            with self.session() as conn:
                conn.execute(
                    """
                    INSERT INTO retrieval_traces (trace_id, conversation_id, message_id, question,
                                                  rewritten_query, variants, candidates, boosts,
                                                  final_pages, rerank_mode, top_cosine,
                                                  first_token_ms, total_ms, mode, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(trace_id) DO UPDATE SET
                        conversation_id=excluded.conversation_id, message_id=excluded.message_id,
                        question=excluded.question, rewritten_query=excluded.rewritten_query,
                        variants=excluded.variants, candidates=excluded.candidates,
                        boosts=excluded.boosts, final_pages=excluded.final_pages,
                        rerank_mode=excluded.rerank_mode, top_cosine=excluded.top_cosine,
                        first_token_ms=excluded.first_token_ms, total_ms=excluded.total_ms,
                        mode=excluded.mode
                    """,
                    (
                        trace.trace_id,
                        trace.conversation_id,
                        trace.message_id,
                        trace.question,
                        trace.rewritten_query,
                        _json(trace.variants),
                        _json(trace.candidates),
                        _json(trace.boosts),
                        _json(trace.final_pages),
                        trace.rerank_mode,
                        float(trace.top_cosine),
                        float(trace.first_token_ms),
                        float(trace.total_ms),
                        trace.mode,
                        _iso(trace.created_at),
                    ),
                )
            logger.info(
                "app.storage.sqlite_manager",
                "写入检索追踪",
                trace_id=trace.trace_id,
                candidates=len(trace.candidates),
                pages=trace.final_pages,
            )
            return 1
        except Exception:
            logger.exception("app.storage.sqlite_manager", "写入检索追踪失败", trace_id=trace.trace_id)
            raise StorageError("写入检索追踪失败", detail={"trace_id": trace.trace_id}) from None

    def get_retrieval_traces(
        self,
        conversation_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 100,
    ) -> list[RetrievalTrace]:
        """按条件查询检索追踪（created_at 倒序）。"""
        try:
            sql = "SELECT * FROM retrieval_traces"
            params: list[Any] = []
            conditions: list[str] = []
            if conversation_id:
                conditions.append("conversation_id=?")
                params.append(conversation_id)
            if trace_id:
                conditions.append("trace_id=?")
                params.append(trace_id)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            with self.session() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [self._row_to_trace(row) for row in rows]
        except Exception:
            logger.exception("app.storage.sqlite_manager", "查询检索追踪失败", conversation_id=conversation_id)
            raise StorageError("查询检索追踪失败") from None

    @staticmethod
    def _row_to_trace(row: sqlite3.Row) -> RetrievalTrace:
        return RetrievalTrace(
            trace_id=row["trace_id"],
            conversation_id=row["conversation_id"],
            message_id=row["message_id"],
            question=row["question"],
            rewritten_query=row["rewritten_query"],
            variants=_unjson(row["variants"], []),
            candidates=_unjson(row["candidates"], []),
            boosts=_unjson(row["boosts"], {}),
            final_pages=_unjson(row["final_pages"], []),
            rerank_mode=row["rerank_mode"],
            top_cosine=float(row["top_cosine"]),
            first_token_ms=float(row["first_token_ms"]),
            total_ms=float(row["total_ms"]),
            mode=row["mode"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )


_manager: SQLiteManager | None = None
_manager_lock = threading.Lock()


def get_sqlite_manager(db_path: Path | str | None = None) -> SQLiteManager:
    """获取进程级单例（传入 db_path 时绕过单例，便于测试隔离）。"""
    global _manager
    if db_path is not None:
        return SQLiteManager(db_path)
    with _manager_lock:
        if _manager is None:
            _manager = SQLiteManager()
    return _manager
