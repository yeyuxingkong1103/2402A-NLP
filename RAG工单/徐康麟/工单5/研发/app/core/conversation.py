# -*- coding: utf-8 -*-
"""工单4 多轮会话（``session_id`` 维度）与指代消解辅助。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

设计依据：设计/接口设计.md §3.3（``session_id`` 多轮）与 §2.2（``--interactive`` 交互式 CLI）。

职责：
    * 记录每个会话的问答轮次（内存 + 可选落盘 ``SqliteManager``）；
    * **指代消解**：问句出现「它 / 其 / 该公司 / 这个项目 / 上述」等指代词且缺少主体时，
      把上一轮的问句主体拼到当前问句前面（**只补主体，不改写事实**，且拼接过的事实必须留痕）；
    * 会话轮数上限由 ``RAG_ANSWER__HISTORY_TURNS`` 控制。

纪律：本模块**不做任何答案生成**；它只决定「检索与生成看到的问题文本是什么」。
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

#: 指代词（出现且问句没有主体名词时触发上文补全）
_ANAPHORA = re.compile(r"(它|其|该|这个|那个|上述|前述|该公司|本项目|这家|此项)")

#: 主体名词（招股书高频主体，用于判断上一轮问句里哪一段可以拿来补全）
_SUBJECT = re.compile(r"[\u4e00-\u9fff]{4,20}(?:股份有限公司|有限公司)")


def _lazy_logger(logger: Any, module: str = "conversation") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class Turn:
    """一轮问答。"""

    question: str
    answer: str
    ts: float
    trace_id: str = ""
    cited_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "ts": self.ts,
                "trace_id": self.trace_id, "cited_pages": self.cited_pages}


class ConversationStore:
    """会话存储（线程安全；容量上限 + 可选落盘）。"""

    def __init__(self, *, max_turns: int = 5, max_sessions: int = 200,
                 storage: Any = None, logger: Any = None) -> None:
        self.max_turns = max(1, int(max_turns))
        self.max_sessions = max(1, int(max_sessions))
        self.storage = storage
        self.log = _lazy_logger(logger)
        self._sessions: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()

    def add(self, session_id: str, turn: Turn) -> None:
        """追加一轮（超上限丢弃最旧）。"""
        if not session_id:
            return
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append(turn)
            if len(turns) > self.max_turns:
                del turns[: len(turns) - self.max_turns]
            if len(self._sessions) > self.max_sessions:
                oldest = min(self._sessions, key=lambda sid: self._sessions[sid][-1].ts)
                if oldest != session_id:
                    self._sessions.pop(oldest, None)
        self.log.log_event("conversation.turn_added", session_id=session_id,
                           turns=len(self._sessions.get(session_id, [])),
                           question_digest=turn.question[:60])
        if self.storage is not None:
            try:
                self.storage.touch_session(session_id, title=turn.question)
            except Exception as exc:  # noqa: BLE001 —— 落盘失败不影响会话
                self.log.log_event("conversation.persist_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id or "", []))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id or "", None)

    def pending_subject(self, session_id: str) -> str | None:
        """取上一轮问句里的主体名词（找不到返回 ``None``）。"""
        turns = self.history(session_id)
        if not turns:
            return None
        match = _SUBJECT.search(turns[-1].question)
        return match.group(0) if match else None

    def resolve(self, question: str, session_id: str | None) -> tuple[str, bool]:
        """指代消解：返回 ``(用于检索/生成的问题, 是否拼接了上文主体)``。"""
        log = self.log
        with log.enter("resolve", {"session_id": session_id, "question_digest": (question or "")[:60]}) as span:
            if not session_id:
                span.set_output({"resolved": False, "reason": "无 session_id（单轮问答）"})
                return question, False
            if not _ANAPHORA.search(question or "") or _SUBJECT.search(question or ""):
                span.set_output({"resolved": False, "reason": "无指代词或问句已含主体"})
                return question, False
            subject = self.pending_subject(session_id)
            if not subject:
                span.set_output({"resolved": False, "reason": "上文未找到主体名词"})
                return question, False
            resolved = f"{subject}{question}"
            log.log_event("conversation.anaphora_resolved", session_id=session_id, subject=subject,
                          resolved_digest=resolved[:80])
            span.set_output({"resolved": True, "subject": subject, "chars": len(resolved)})
            return resolved, True


def now_turn(question: str, answer: str, *, trace_id: str = "", cited_pages: list[int] | None = None) -> Turn:
    """构造一轮记录。"""
    return Turn(question=question, answer=answer, ts=time.time(), trace_id=trace_id,
                cited_pages=list(cited_pages or []))
