# -*- coding: utf-8 -*-
"""工单3 多轮对话存储（设计/接口设计.md §2.4、§3.20 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

窗口：最近 **5 轮（10 条消息）**注入 prompt；数据库保留全量。
表：``conversations`` / ``messages``（DDL 见设计 §5.3，复用 ``pdf_parser.connect_sqlite`` 的同一套 DDL）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field, fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .citation import Citation
from .config import AppConfig, get_config
from .errors import StorageError, wrap

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"


def _lazy_logger(logger: Any, module: str = "conversation") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class Turn:
    """一轮消息（用户或助手）。"""

    session_id: str
    role: str                                  # "user" | "assistant"
    content: str
    created_at: str
    citations: list[Citation] = field(default_factory=list)
    first_token_ms: float | None = None
    total_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citations"] = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in self.citations]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Turn":
        names = {f.name for f in dc_fields(cls)}
        payload = {k: v for k, v in data.items() if k in names}
        payload["citations"] = [Citation.from_dict(c) for c in (data.get("citations") or [])]
        return cls(**payload)


class ConversationStore:
    """会话与消息的 SQLite 存储（线程安全：每次操作独立连接 + 事务）。"""

    def __init__(self, *, db_path: Path | str | None = None, cfg: AppConfig | None = None,
                 logger: Any = None) -> None:
        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)
        self.db_path = Path(db_path) if db_path is not None else self.cfg.paths.index_dir / "rag.sqlite3"
        self._lock = threading.RLock()
        self._ensure_schema()

    # -- 内部 ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """打开连接并确保 schema 存在（幂等）。"""
        from .pdf_parser import connect_sqlite

        return connect_sqlite(self.db_path)

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.close()

    # -- 会话 ------------------------------------------------------------
    def create_session(self, *, session_id: str | None = None) -> str:
        """创建会话（未给 id 时自动生成），返回 session_id。"""
        with self.log.enter("ConversationStore.create_session", {"session_id": session_id}) as span:
            sid = session_id or f"s{int(time.time() * 1000) % 100000000:08d}"
            now = _now_iso()
            conn = self._connect()
            try:
                with self._lock, conn:
                    conn.execute(
                        "INSERT INTO conversations(session_id,created_at,updated_at,message_count)"
                        " VALUES(?,?,?,0) ON CONFLICT(session_id) DO NOTHING", (sid, now, now),
                    )
            except sqlite3.Error as exc:
                self.log.log_event("conversation.error", level="ERROR", op="create_session", session_id=sid,
                                   error_type=type(exc).__name__, message=str(exc))
                raise wrap(exc, code="RAG-7000", stage="storage", session_id=sid) from exc
            finally:
                conn.close()
            span.set_output({"session_id": sid})
            return sid

    def append_turn(self, session_id: str, turn: Turn) -> None:
        """追加一条消息并更新会话计数（同一事务）。"""
        with self.log.enter("ConversationStore.append_turn",
                            {"session_id": session_id, "role": turn.role, "chars": len(turn.content)}) as span:
            self.create_session(session_id=session_id)
            conn = self._connect()
            try:
                with self._lock, conn:
                    conn.execute(
                        "INSERT INTO messages(session_id,role,content,citations_json,first_token_ms,total_ms,"
                        "created_at) VALUES(?,?,?,?,?,?,?)",
                        (session_id, turn.role, turn.content,
                         json.dumps([c.to_dict() for c in turn.citations], ensure_ascii=False),
                         turn.first_token_ms, turn.total_ms, turn.created_at or _now_iso()),
                    )
                    conn.execute(
                        "UPDATE conversations SET updated_at=?, message_count=message_count+1"
                        " WHERE session_id=?", (_now_iso(), session_id),
                    )
            except sqlite3.Error as exc:
                self.log.log_event("conversation.error", level="ERROR", op="append_turn", session_id=session_id,
                                   error_type=type(exc).__name__, message=str(exc))
                raise wrap(exc, code="RAG-7000", stage="storage", session_id=session_id) from exc
            finally:
                conn.close()
            count = self.message_count(session_id)
            self.log.log_event("conversation.append", session_id=session_id, role=turn.role,
                               chars=len(turn.content), count=count)
            span.set_output({"session_id": session_id, "message_count": count})

    def history(self, session_id: str, *, last_n: int = 5) -> list[Turn]:
        """返回最近 ``last_n`` **轮**的消息（1 轮 = user+assistant，共最多 2*last_n 条，按时间正序）。"""
        with self.log.enter("ConversationStore.history",
                            {"session_id": session_id, "last_n": last_n}) as span:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT role,content,citations_json,first_token_ms,total_ms,created_at FROM messages"
                    " WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, max(int(last_n), 1) * 2),
                ).fetchall()
            except sqlite3.Error as exc:
                self.log.log_event("conversation.error", level="ERROR", op="history", session_id=session_id,
                                   error_type=type(exc).__name__, message=str(exc))
                raise wrap(exc, code="RAG-7000", stage="storage", session_id=session_id) from exc
            finally:
                conn.close()
            turns = [
                Turn(session_id=session_id, role=row[0], content=row[1],
                     citations=[Citation.from_dict(c) for c in json.loads(row[2] or "[]")],
                     first_token_ms=row[3], total_ms=row[4], created_at=row[5])
                for row in reversed(rows)
            ]
            self.log.log_event("conversation.history", session_id=session_id, last_n=last_n,
                               returned=len(turns))
            span.set_output({"session_id": session_id, "returned": len(turns)})
            return turns

    def list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """列出会话（按最近更新倒序）。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT session_id,created_at,updated_at,message_count FROM conversations"
                " ORDER BY updated_at DESC LIMIT ?", (max(int(limit), 1),),
            ).fetchall()
        except sqlite3.Error as exc:
            raise wrap(exc, code="RAG-7000", stage="storage", op="list_sessions") from exc
        finally:
            conn.close()
        return [{"session_id": r[0], "created_at": r[1], "updated_at": r[2], "message_count": r[3]}
                for r in rows]

    def clear(self, session_id: str) -> int:
        """清空会话消息，返回删除条数。"""
        conn = self._connect()
        try:
            with self._lock, conn:
                cur = conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
                deleted = int(cur.rowcount or 0)
                conn.execute("UPDATE conversations SET message_count=0, updated_at=? WHERE session_id=?",
                             (_now_iso(), session_id))
        except sqlite3.Error as exc:
            raise wrap(exc, code="RAG-7000", stage="storage", session_id=session_id) from exc
        finally:
            conn.close()
        self.log.log_event("conversation.clear", session_id=session_id, deleted=deleted)
        return deleted

    def message_count(self, session_id: str) -> int:
        """会话消息条数（会话不存在返回 0）。"""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()
        except sqlite3.Error as exc:
            raise wrap(exc, code="RAG-7000", stage="storage", session_id=session_id) from exc
        finally:
            conn.close()
        return int(row[0]) if row else 0

    def to_dict(self) -> dict[str, Any]:
        return {"db_path": str(self.db_path), "sessions": len(self.list_sessions(limit=1000))}


_STORE: ConversationStore | None = None


def get_conversation_store(*, cfg: AppConfig | None = None, logger: Any = None) -> ConversationStore:
    """进程级单例（同一库文件复用连接配置）。"""
    global _STORE
    if _STORE is None:
        _STORE = ConversationStore(cfg=cfg, logger=logger)
    return _STORE
