# -*- coding: utf-8 -*-
"""工单4 轻量持久化（``sqlite3`` 标准库）：问答历史、会话、答案级缓存。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

设计依据：
    * 设计/接口设计.md §4.2（``session_id`` 多轮）与 §5.3（留痕契约）；
    * 任务书性能策略：「问题级与答案级缓存」——缓存落盘后**重启进程仍然命中**；
    * 纪律：**不修改 T4 的 ``研发/data/index/rag.sqlite3``**，问答侧使用独立库文件
      ``研发/data/index/qa_runtime.sqlite3``（T4 的库用于知识存储，职责不同，互不覆盖）。

失败策略：任何一次落盘失败都**只写 WARNING 并继续**（问答不能被持久化拖死），
但绝不允许静默：日志里必须看得到 ``storage.*`` 事件与异常类型。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.errors import StorageError

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_history (
    trace_id      TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    session_id    TEXT,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    refused       INTEGER NOT NULL DEFAULT 0,
    refusal_reason TEXT,
    citations     TEXT,
    cited_pages   TEXT,
    latency_ms    REAL,
    first_token_ms REAL,
    backend       TEXT,
    model         TEXT,
    cache_hit     INTEGER NOT NULL DEFAULT 0,
    language      TEXT,
    top_k         INTEGER
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    turns       INTEGER NOT NULL DEFAULT 0,
    title       TEXT
);
CREATE TABLE IF NOT EXISTS answer_cache (
    cache_key  TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_qa_history_ts ON qa_history(ts DESC);
CREATE INDEX IF NOT EXISTS idx_qa_history_session ON qa_history(session_id, ts DESC);
"""


def _lazy_logger(logger: Any, module: str = "sqlite_manager") -> Any:
    if logger is not None:
        return logger
    from ..core.logging_conf import get_logger

    return get_logger(module)


def cache_key(question: str, *, lang: str = "auto", top_k: int = 5) -> str:
    """答案级缓存键：问句归一化 + 语言 + Top-K（**不含**会话，保证跨会话复用）。"""
    normalized = "".join(str(question or "").split()).lower()
    raw = f"{normalized}|{lang}|{top_k}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class HistoryRow:
    """一条问答历史。"""

    trace_id: str
    ts: float
    question: str
    answer: str
    refused: bool
    cited_pages: list[int]
    latency_ms: float
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "ts": self.ts, "question": self.question,
                "answer": self.answer, "refused": self.refused, "cited_pages": self.cited_pages,
                "latency_ms": self.latency_ms, "cache_hit": self.cache_hit}


class SqliteManager:
    """问答运行时持久化（线程安全：单连接 + 互斥锁 + ``check_same_thread=False``）。"""

    def __init__(self, db_path: Path | str, *, logger: Any = None) -> None:
        self.db_path = Path(db_path)
        self.log = _lazy_logger(logger)
        self._lock = threading.Lock()
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.executescript(_SCHEMA)
                self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"问答运行时库初始化失败：{self.db_path}（{exc}）",
                               detail={"db": str(self.db_path)}) from exc
        self.log.log_event("storage.ready", db=str(self.db_path))

    # -- 写入 ------------------------------------------------------------
    def record_qa(self, *, trace_id: str, question: str, answer: str, refused: bool,
                  session_id: str | None = None, refusal_reason: str | None = None,
                  citations: Sequence[Mapping[str, Any]] | None = None,
                  cited_pages: Sequence[int] | None = None, latency_ms: float = 0.0,
                  first_token_ms: float = 0.0, backend: str = "", model: str = "",
                  cache_hit: bool = False, language: str = "zh", top_k: int = 5) -> None:
        """落盘一次问答（失败只 WARNING，不阻断问答）。"""
        started = time.perf_counter()
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO qa_history (trace_id, ts, session_id, question, answer,"
                    " refused, refusal_reason, citations, cited_pages, latency_ms, first_token_ms,"
                    " backend, model, cache_hit, language, top_k)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (trace_id, time.time(), session_id, question, answer, int(bool(refused)),
                     refusal_reason, json.dumps(list(citations or []), ensure_ascii=False),
                     json.dumps(list(cited_pages or [])), float(latency_ms), float(first_token_ms),
                     backend, model, int(bool(cache_hit)), language, int(top_k)),
                )
                self._conn.commit()
            self.log.log_event("storage.qa_recorded", trace_id=trace_id, refused=bool(refused),
                               chars=len(answer), elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
        except sqlite3.Error as exc:
            self.log.log_event("storage.write_failed", level="WARNING", trace_id=trace_id,
                               error=f"{type(exc).__name__}: {exc}", action="问答继续，仅历史缺失")

    def touch_session(self, session_id: str, *, title: str = "") -> None:
        """更新会话计数（失败只 WARNING）。"""
        if not session_id:
            return
        now = time.time()
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO sessions (session_id, created_at, updated_at, turns, title)"
                    " VALUES (?,?,?,1,?) ON CONFLICT(session_id) DO UPDATE SET"
                    " updated_at=excluded.updated_at, turns=turns+1,"
                    " title=CASE WHEN sessions.title='' THEN excluded.title ELSE sessions.title END",
                    (session_id, now, now, title[:120]),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            self.log.log_event("storage.session_failed", level="WARNING", session_id=session_id,
                               error=f"{type(exc).__name__}: {exc}")

    def save_answer_cache(self, key: str, question: str, payload: Mapping[str, Any]) -> None:
        """写答案级缓存（失败只 WARNING）。"""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO answer_cache (cache_key, question, payload, created_at, hits)"
                    " VALUES (?,?,?,?, COALESCE((SELECT hits FROM answer_cache WHERE cache_key=?),0))",
                    (key, question, json.dumps(payload, ensure_ascii=False), time.time(), key),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            self.log.log_event("storage.cache_write_failed", level="WARNING", key=key,
                               error=f"{type(exc).__name__}: {exc}")

    def bump_cache_hit(self, key: str) -> None:
        """缓存命中计数（失败只 WARNING）。"""
        try:
            with self._lock:
                self._conn.execute("UPDATE answer_cache SET hits = hits + 1 WHERE cache_key = ?", (key,))
                self._conn.commit()
        except sqlite3.Error as exc:
            self.log.log_event("storage.cache_bump_failed", level="WARNING", key=key,
                               error=f"{type(exc).__name__}: {exc}")

    # -- 读取 ------------------------------------------------------------
    def load_answer_cache(self, key: str) -> dict[str, Any] | None:
        """读答案级缓存（不存在返回 ``None``）。"""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT payload FROM answer_cache WHERE cache_key = ?", (key,)).fetchone()
        except sqlite3.Error as exc:
            self.log.log_event("storage.cache_read_failed", level="WARNING", key=key,
                               error=f"{type(exc).__name__}: {exc}")
            return None
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except (TypeError, ValueError) as exc:  # 显式降级：坏缓存当作未命中
            self.log.log_event("storage.cache_corrupt", level="WARNING", key=key,
                               error=f"{type(exc).__name__}: {exc}")
            return None

    def recent(self, limit: int = 20) -> list[HistoryRow]:
        """最近问答历史（倒序）。"""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT trace_id, ts, question, answer, refused, cited_pages, latency_ms, cache_hit"
                    " FROM qa_history ORDER BY ts DESC LIMIT ?", (int(limit),)).fetchall()
        except sqlite3.Error as exc:
            self.log.log_event("storage.recent_failed", level="WARNING",
                               error=f"{type(exc).__name__}: {exc}")
            return []
        out: list[HistoryRow] = []
        for row in rows:
            try:
                pages = json.loads(row["cited_pages"] or "[]")
            except (TypeError, ValueError):
                pages = []
            out.append(HistoryRow(trace_id=row["trace_id"], ts=float(row["ts"]), question=row["question"],
                                  answer=row["answer"], refused=bool(row["refused"]),
                                  cited_pages=pages, latency_ms=float(row["latency_ms"] or 0.0),
                                  cache_hit=bool(row["cache_hit"])))
        return out

    def stats(self) -> dict[str, Any]:
        """库内统计（供 /health 与报告）。"""
        try:
            with self._lock:
                total = self._conn.execute("SELECT COUNT(*) AS n FROM qa_history").fetchone()["n"]
                refused = self._conn.execute("SELECT COUNT(*) AS n FROM qa_history WHERE refused=1").fetchone()["n"]
                cached = self._conn.execute("SELECT COUNT(*) AS n FROM answer_cache").fetchone()["n"]
                sessions = self._conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
            return {"db": str(self.db_path), "qa_total": int(total), "refused_total": int(refused),
                    "cache_entries": int(cached), "sessions": int(sessions)}
        except sqlite3.Error as exc:
            self.log.log_event("storage.stats_failed", level="WARNING",
                               error=f"{type(exc).__name__}: {exc}")
            return {"db": str(self.db_path), "error": f"{type(exc).__name__}: {exc}"}

    def close(self) -> None:
        """关闭连接（幂等）。"""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error as exc:  # 显式降级：关不掉也要留痕
                self.log.log_event("storage.close_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")
