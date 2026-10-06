# -*- coding: utf-8 -*-
"""工单3 SQLite 存储层（设计/接口设计.md §3.21 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

职责：
    * 统一 DDL（复用 ``pdf_parser.SQLITE_DDL``，新增 ``feedback`` 表用于点赞/点踩，§5.3 允许新增表）；
    * ``documents/pages/tables/chunks`` 的 UPSERT 与回查（引用校验规则②的底座）；
    * ``retrieval_traces`` / ``runs`` / ``feedback`` 的记录；
    * 所有异常包装为 ``StorageError`` 抛出（不吞），入口/出口结构化日志。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.chunker import Chunk
from ..core.config import AppConfig, get_config
from ..core.errors import StorageError
from ..core.pdf_parser import SQLITE_DDL, connect_sqlite
from ..core.text_utils import text_digest

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 反馈表（点赞/点踩）：UI 的唯一持久化出口；message_id 允许为空（非会话式的一次问答也允许投票）
FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, message_id INTEGER,
  answer_id TEXT, trace_id TEXT, rating TEXT NOT NULL, comment TEXT,
  created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id, id);
"""


def _lazy_logger(logger: Any, module: str = "sqlite_manager") -> Any:
    if logger is not None:
        return logger
    from ..core.logging_conf import get_logger

    return get_logger(module)


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class SQLiteManager:
    """SQLite 存储管理器（线程安全：每线程独立连接 + WAL）。"""

    def __init__(self, *, db_path: Path | str | None = None, cfg: AppConfig | None = None,
                 logger: Any = None) -> None:
        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)
        self.db_path = Path(db_path) if db_path is not None else self.cfg.paths.index_dir / "rag.sqlite3"
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self.init_schema()

    # -- 连接 ------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        """取本线程连接（首次创建时建 schema 并设 PRAGMA）。"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = connect_sqlite(self.db_path)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA foreign_keys=ON")
                self._local.conn = conn
            except sqlite3.Error as exc:
                raise StorageError(f"SQLite 连接失败：{self.db_path}（{exc}）",
                                   detail={"db_path": str(self.db_path)}) from exc
        return conn

    def init_schema(self) -> None:
        """建表（幂等）。"""
        with self.log.enter("SQLiteManager.init_schema", {"db_path": str(self.db_path)}) as span:
            try:
                conn = self._conn()
                with self._write_lock, conn:
                    conn.executescript(SQLITE_DDL)
                    conn.executescript(FEEDBACK_DDL)
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            except sqlite3.Error as exc:
                raise StorageError(f"建表失败：{self.db_path}（{exc}）") from exc
            self.log.log_event("sqlite.init_schema", tables=len(tables), names=tables)
            span.set_output({"tables": len(tables)})

    # -- UPSERT ----------------------------------------------------------
    def _upsert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> int:
        """按主键 UPSERT 一批行（列名取自 rows 的键，缺列不写）。"""
        if not rows:
            return 0
        started = time.perf_counter()
        columns = list(rows[0].keys())
        placeholders = ",".join("?" for _ in columns)
        sql = (f"INSERT INTO {table}({','.join(columns)}) VALUES({placeholders}) "
               f"ON CONFLICT DO UPDATE SET " + ",".join(f"{c}=excluded.{c}" for c in columns))
        try:
            conn = self._conn()
            with self._write_lock, conn:
                conn.executemany(sql, [[row.get(c) for c in columns] for row in rows])
        except sqlite3.Error as exc:
            raise StorageError(f"{table} UPSERT 失败（{len(rows)} 行）：{exc}",
                               detail={"table": table}) from exc
        ms = round((time.perf_counter() - started) * 1000, 2)
        self.log.log_event("sqlite.upsert", table=table, rows=len(rows), ms=ms)
        return len(rows)

    def upsert_documents(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """写 ``documents``。"""
        with self.log.enter("SQLiteManager.upsert_documents", {"rows": len(rows)}) as span:
            count = self._upsert("documents", rows)
            span.set_output({"rows": count})
            return count

    def upsert_pages(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """写 ``pages``。"""
        with self.log.enter("SQLiteManager.upsert_pages", {"rows": len(rows)}) as span:
            count = self._upsert("pages", rows)
            span.set_output({"rows": count})
            return count

    def upsert_tables(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """写 ``tables``。"""
        with self.log.enter("SQLiteManager.upsert_tables", {"rows": len(rows)}) as span:
            count = self._upsert("tables", rows)
            span.set_output({"rows": count})
            return count

    def upsert_chunks(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """写 ``chunks``。"""
        with self.log.enter("SQLiteManager.upsert_chunks", {"rows": len(rows)}) as span:
            count = self._upsert("chunks", rows)
            span.set_output({"rows": count})
            return count

    # -- 查询 ------------------------------------------------------------
    def fetch_chunk(self, chunk_id: str) -> Chunk | None:
        """按 chunk_id 取块（引用校验规则②）。"""
        with self.log.enter("SQLiteManager.fetch_chunk", {"chunk_id": chunk_id}) as span:
            row = self._conn().execute(
                "SELECT chunk_id,file_name,page,page_start,page_end,type,section,content,table_id,char_count,ord"
                " FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
            chunk = None if row is None else Chunk(
                chunk_id=row[0], file_name=row[1], page=int(row[2]), page_start=int(row[3]),
                page_end=int(row[4]), type=row[5], section=row[6] or "", content=row[7] or "",
                keywords=[], table_id=row[8], char_count=int(row[9]), order=int(row[10]))
            self.log.log_event("sqlite.query", op="fetch_chunk", rows=0 if chunk is None else 1)
            span.set_output({"found": chunk is not None})
            return chunk

    def fetch_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        """批量取块（一次查询，避免 N+1）。"""
        ids = list(chunk_ids)
        with self.log.enter("SQLiteManager.fetch_chunks", {"chunk_ids": len(ids)}) as span:
            if not ids:
                span.set_output({"found": 0})
                return {}
            marks = ",".join("?" for _ in ids)
            rows = self._conn().execute(
                "SELECT chunk_id,file_name,page,page_start,page_end,type,section,content,table_id,char_count,ord"
                f" FROM chunks WHERE chunk_id IN ({marks})", ids).fetchall()
            out: dict[str, Chunk] = {}
            for row in rows:
                out[row[0]] = Chunk(chunk_id=row[0], file_name=row[1], page=int(row[2]),
                                    page_start=int(row[3]), page_end=int(row[4]), type=row[5],
                                    section=row[6] or "", content=row[7] or "", keywords=[],
                                    table_id=row[8], char_count=int(row[9]), order=int(row[10]))
            self.log.log_event("sqlite.query", op="fetch_chunks", args_digest=text_digest(",".join(ids)),
                               rows=len(out))
            span.set_output({"found": len(out)})
            return out

    def count_chunks(self, *, file_name: str | None = None, type: str | None = None) -> int:
        """统计块数（可按文件/类型过滤）。"""
        with self.log.enter("SQLiteManager.count_chunks", {"file_name": file_name, "type": type}) as span:
            sql, args = "SELECT COUNT(*) FROM chunks WHERE 1=1", []
            if file_name:
                sql += " AND file_name=?"
                args.append(file_name)
            if type:
                sql += " AND type=?"
                args.append(type)
            count = int(self._conn().execute(sql, args).fetchone()[0])
            self.log.log_event("sqlite.query", op="count_chunks", rows=1)
            span.set_output({"count": count})
            return count

    def page_count(self, file_name: str) -> int | None:
        """取文件页数（引用页码合法性判据）。"""
        with self.log.enter("SQLiteManager.page_count", {"file_name": file_name}) as span:
            row = self._conn().execute("SELECT page_count FROM documents WHERE file_name=?",
                                       (file_name,)).fetchone()
            span.set_output({"page_count": None if row is None else int(row[0])})
            return None if row is None else int(row[0])

    def list_documents(self) -> list[dict[str, Any]]:
        """列出已入库文件（供 ``QAEngine.files()`` 拼装，禁止硬编码文件名）。"""
        rows = self._conn().execute(
            "SELECT file_name,size_bytes,sha256_16,page_count,parsed_at FROM documents ORDER BY file_name"
        ).fetchall()
        return [{"file_name": r[0], "size_bytes": int(r[1]), "sha256_16": r[2], "page_count": int(r[3]),
                 "parsed_at": r[4]} for r in rows]

    # -- 记录 ------------------------------------------------------------
    def record_retrieval(self, payload: Mapping[str, Any]) -> None:
        """记录检索轨迹（``retrieval_traces``）。"""
        with self.log.enter("SQLiteManager.record_retrieval",
                            {"trace_id": payload.get("trace_id"), "top_k": payload.get("top_k")}) as span:
            row = {
                "trace_id": payload.get("trace_id"), "session_id": payload.get("session_id"),
                "question": payload.get("question", ""), "rewritten_query": payload.get("rewritten_query"),
                "file_names": json.dumps(payload.get("file_names"), ensure_ascii=False),
                "top_k": int(payload.get("top_k") or 0),
                "chunk_ids_json": json.dumps(payload.get("chunk_ids") or [], ensure_ascii=False),
                "scores_json": json.dumps(payload.get("scores") or [], ensure_ascii=False),
                "stages_json": json.dumps(payload.get("stages") or {}, ensure_ascii=False),
                "created_at": payload.get("created_at") or _now_iso(),
            }
            self._upsert("retrieval_traces", [row])
            self.log.log_event("sqlite.query", op="record_retrieval", rows=1)
            span.set_output({"chunk_ids": len(payload.get("chunk_ids") or [])})

    def record_run(self, *, run_id: str, kind: str, stats: Mapping[str, Any], ok: bool,
                   started_at: str | None = None) -> None:
        """记录一次运行（``runs``）。"""
        with self.log.enter("SQLiteManager.record_run", {"run_id": run_id, "kind": kind, "ok": ok}) as span:
            self._upsert("runs", [{"run_id": run_id, "kind": kind,
                                   "started_at": started_at or _now_iso(), "finished_at": _now_iso(),
                                   "stats_json": json.dumps(dict(stats), ensure_ascii=False),
                                   "ok": 1 if ok else 0}])
            span.set_output({"ok": ok})

    def record_feedback(self, *, rating: str, session_id: str | None = None, message_id: int | None = None,
                        answer_id: str | None = None, trace_id: str | None = None,
                        comment: str | None = None) -> int:
        """记录点赞/点踩（``feedback``）。``rating`` 只接受 ``up``/``down``。"""
        if rating not in {"up", "down"}:
            raise StorageError(f"非法 rating：{rating}（只接受 up/down）", detail={"rating": rating})
        with self.log.enter("SQLiteManager.record_feedback",
                            {"rating": rating, "session_id": session_id, "message_id": message_id}) as span:
            conn = self._conn()
            try:
                with self._write_lock, conn:
                    cur = conn.execute(
                        "INSERT INTO feedback(session_id,message_id,answer_id,trace_id,rating,comment,created_at)"
                        " VALUES(?,?,?,?,?,?,?)",
                        (session_id, message_id, answer_id, trace_id, rating, comment, _now_iso()))
                feedback_id = int(cur.lastrowid)
            except sqlite3.Error as exc:
                raise StorageError(f"写 feedback 失败：{exc}") from exc
            self.log.log_event("sqlite.query", op="record_feedback", rows=1, rating=rating)
            span.set_output({"feedback_id": feedback_id})
            return feedback_id

    def list_feedback(self, *, session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """列出反馈（UI 回显用）。"""
        sql = ("SELECT id,session_id,message_id,answer_id,trace_id,rating,comment,created_at FROM feedback")
        args: list[Any] = []
        if session_id:
            sql += " WHERE session_id=?"
            args.append(session_id)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        rows = self._conn().execute(sql, args).fetchall()
        return [{"id": r[0], "session_id": r[1], "message_id": r[2], "answer_id": r[3], "trace_id": r[4],
                 "rating": r[5], "comment": r[6], "created_at": r[7]} for r in rows]

    def close(self) -> None:
        """关闭本线程连接（幂等）。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def to_dict(self) -> dict[str, Any]:
        """摘要（健康检查/日志用）。"""
        return {"db_path": str(self.db_path), "exists": self.db_path.exists()}


_instance: SQLiteManager | None = None
_instance_lock = threading.Lock()


def get_sqlite_manager(*, db_path: Path | str | None = None, cfg: AppConfig | None = None,
                       logger: Any = None) -> SQLiteManager:
    """进程级单例（``db_path`` 非空时新建独立实例，便于测试隔离）。"""
    global _instance
    if db_path is not None:
        return SQLiteManager(db_path=db_path, cfg=cfg, logger=logger)
    with _instance_lock:
        if _instance is None:
            _instance = SQLiteManager(cfg=cfg, logger=logger)
        return _instance
