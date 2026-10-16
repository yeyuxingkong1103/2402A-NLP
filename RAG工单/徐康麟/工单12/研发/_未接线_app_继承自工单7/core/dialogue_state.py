# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

多轮对话状态（DialogueState / DialogueStateStore）。

## 与工单4 ``conversation.py`` 的区别（这是工单5 的核心增量）
工单4 的 ``ConversationStore.resolve`` 只能做一件事：**当上一轮问句里出现了公司全称、
且当前问句含指代词时，把那个公司名拼到当前问句前面**。它把「主体」存在**上一轮的问句文本**里，
因此有两个硬伤（实测）：

1. **焦点会丢**：Q2「他参与的哪个工程荣获了国家科技进步一等奖？」的问句里**没有**公司名，
   于是 Q3「这个公司的法定代表人是谁？」取不到主体 → 消解失败。
2. **无法换主体**：Q4「那武汉力源信息技术股份有限公司呢？」是**省略式**（只有实体、没有问题），
   工单4 既不认「那…呢？」也不保留「上一轮问的是什么字段」→ 无法补全。

工单5 改为**状态化**：显式维护
    * ``focus_entity``  —— 当前谈论的公司全称（焦点实体），与问句文本解耦；
    * ``last_slot``     —— 上一轮问的「槽位模板」（把实体与指代词挖掉后的问句骨架），
                           供「那 X 呢？」这类省略式继承；
    * ``entity_stack``  —— 出现过的实体栈（最近的在末尾），支持「回到上一家」；
    * ``turns``         —— 最近 N 轮（默认 3，按工单「速度优先」要求）问答记录。

线程安全；可选按 session 落盘（``persist_dir``），进程重启后仍能续聊。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"


def _lazy_logger(logger: Any, module: str = "dialogue_state") -> Any:
    """延迟取 logger，避免与 logging_conf 循环导入。"""
    if logger is not None:
        return logger
    try:
        from .logging_conf import get_logger

        return get_logger(module)
    except Exception:  # noqa: BLE001 —— 日志不可用时不得阻断对话
        return _NullLogger()


class _NullLogger:
    """日志后端不可用时的空实现（保证对话链路永不因日志而失败）。"""

    def log_event(self, *_a: Any, **_kw: Any) -> None:
        return None

    def bind(self, **_kw: Any) -> "_NullLogger":
        return self

    class _Span:
        def set_output(self, *_a: Any, **_kw: Any) -> None:
            return None

        def __enter__(self) -> "_NullLogger._Span":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

    def enter(self, *_a: Any, **_kw: Any) -> "_NullLogger._Span":
        return _NullLogger._Span()


@dataclass(slots=True)
class TurnRecord:
    """一轮问答的完整记录（原句 / 改写后独立问句 / 答案 / 焦点）。"""

    question: str                 # 用户原始问句
    rewritten_query: str          # 改写后的独立可检索问句
    answer: str
    ts: float = field(default_factory=time.time)
    trace_id: str = ""
    cited_pages: list[int] = field(default_factory=list)
    focus_entity: str | None = None
    strategy: str = "none"
    refused: bool = False
    latency_ms: float = 0.0
    slot: str | None = None       # 本轮问句的槽位模板

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DialogueState:
    """一个会话的多轮状态。"""

    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    #: 焦点实体（当前谈论的公司全称）
    focus_entity: str | None = None
    #: 上一轮的「槽位模板」，如「法定代表人是谁？」
    last_slot: str | None = None
    #: 上一轮的独立问句（改写后）
    last_query: str | None = None
    #: 实体栈（最近在末尾）
    entity_stack: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---------------------------------------------------------------- 便捷访问
    def recent(self, n: int = 3) -> list[TurnRecord]:
        """最近 n 轮（时间正序）。"""
        return list(self.turns[-max(1, int(n)):])

    @property
    def turn_count(self) -> int:
        """累计轮次（不受保留窗口限制，用于 ``turn_index``）。"""
        return int(self.total_turns)

    #: 累计轮次（turns 会被裁剪，故单独计数）
    total_turns: int = 0

    def push_entity(self, entity: str) -> None:
        """把实体压入栈并设为焦点（去重，最近的在末尾）。"""
        name = (entity or "").strip()
        if not name:
            return
        if name in self.entity_stack:
            self.entity_stack.remove(name)
        self.entity_stack.append(name)
        if len(self.entity_stack) > 8:
            del self.entity_stack[: len(self.entity_stack) - 8]
        self.focus_entity = name

    def previous_entity(self) -> str | None:
        """上一个实体（用于「回到上一家」这类追问，暂未在规则中使用，留作扩展）。"""
        return self.entity_stack[-2] if len(self.entity_stack) >= 2 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "focus_entity": self.focus_entity,
            "last_slot": self.last_slot,
            "last_query": self.last_query,
            "entity_stack": list(self.entity_stack),
            "total_turns": int(self.total_turns),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DialogueState":
        turns = [TurnRecord(**{k: v for k, v in t.items() if k in TurnRecord.__slots__})
                 for t in (data.get("turns") or [])]
        state = cls(
            session_id=str(data.get("session_id") or ""),
            turns=turns,
            focus_entity=data.get("focus_entity"),
            last_slot=data.get("last_slot"),
            last_query=data.get("last_query"),
            entity_stack=list(data.get("entity_stack") or []),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )
        state.total_turns = int(data.get("total_turns") or len(turns))
        return state


class DialogueStateStore:
    """会话状态仓库（线程安全；容量上限 + 可选落盘）。"""

    def __init__(self, *, max_turns: int = 3, max_sessions: int = 200,
                 storage: Any = None, logger: Any = None,
                 persist_dir: Path | str | None = None) -> None:
        self.max_turns = max(1, int(max_turns))
        self.max_sessions = max(1, int(max_sessions))
        self.storage = storage
        self.log = _lazy_logger(logger)
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir is not None:
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001 —— 落盘不可用即退化为纯内存
                self.log.log_event("dialogue.persist_unavailable", level="WARNING",
                                   dir=str(self.persist_dir), error=f"{type(exc).__name__}: {exc}")
                self.persist_dir = None
        self._sessions: dict[str, DialogueState] = {}
        self._lock = threading.RLock()
        self._stats = {"sessions": 0, "turns": 0, "resets": 0}

    # ------------------------------------------------------------------ 基础
    def _path_of(self, session_id: str) -> Path | None:
        if self.persist_dir is None or not session_id:
            return None
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in session_id)[:80]
        return self.persist_dir / f"{safe or 'default'}.json"

    def _load(self, session_id: str) -> DialogueState | None:
        path = self._path_of(session_id)
        if path is None or not path.is_file():
            return None
        try:
            return DialogueState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001 —— 损坏的会话文件不得阻断对话
            self.log.log_event("dialogue.load_failed", level="WARNING", session_id=session_id,
                               error=f"{type(exc).__name__}: {exc}")
            return None

    def _save(self, state: DialogueState) -> None:
        path = self._path_of(state.session_id)
        if path is None:
            return
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001 —— 落盘失败不影响内存会话
            self.log.log_event("dialogue.save_failed", level="WARNING",
                               session_id=state.session_id, error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ 主接口
    def get(self, session_id: str) -> DialogueState:
        """取（或新建）会话状态。"""
        sid = session_id or "default"
        with self._lock:
            state = self._sessions.get(sid)
            if state is None:
                state = self._load(sid) or DialogueState(session_id=sid)
                self._evict_if_needed()
                self._sessions[sid] = state
                self._stats["sessions"] = len(self._sessions)
            return state

    def _evict_if_needed(self) -> None:
        """超过会话上限时淘汰最久未更新的会话（**调用方须已持锁**）。"""
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].updated_at)
            self._sessions.pop(oldest, None)

    def append(self, session_id: str, turn: TurnRecord) -> DialogueState:
        """追加一轮并同步焦点/槽位（超上限丢弃最旧轮）。"""
        state = self.get(session_id)
        with self._lock:
            state.turns.append(turn)
            if len(state.turns) > self.max_turns:
                del state.turns[: len(state.turns) - self.max_turns]
            state.total_turns = max(int(state.total_turns) + 1, len(state.turns))
            state.last_query = turn.rewritten_query
            if turn.slot:
                state.last_slot = turn.slot
            if turn.focus_entity:
                state.push_entity(turn.focus_entity)
            state.updated_at = time.time()
            self._stats["turns"] += 1
            self._save(state)
        self.log.log_event("dialogue.turn_added", session_id=state.session_id,
                           turn=int(state.total_turns), kept=len(state.turns),
                           strategy=turn.strategy, focus=turn.focus_entity,
                           question_digest=turn.question[:60])
        if self.storage is not None:
            try:
                self.storage.touch_session(state.session_id, title=turn.question)
            except Exception as exc:  # noqa: BLE001 —— 落库失败不影响会话
                self.log.log_event("dialogue.persist_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")
        return state

    def reset(self, session_id: str) -> None:
        """清空某个会话。"""
        sid = session_id or "default"
        with self._lock:
            self._sessions.pop(sid, None)
            self._stats["resets"] += 1
            path = self._path_of(sid)
        if path is not None and path.is_file():
            try:
                path.unlink()
            except Exception:  # noqa: BLE001 —— 删除失败不影响内存已清空的事实
                pass
        self.log.log_event("dialogue.reset", session_id=sid)

    def history(self, session_id: str) -> list[TurnRecord]:
        return self.get(session_id).recent(self.max_turns)

    def sessions(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self._stats, "max_turns": self.max_turns,
                    "live_sessions": len(self._sessions),
                    "persist_dir": str(self.persist_dir) if self.persist_dir else None}


__all__ = ["WORK_ORDER", "TurnRecord", "DialogueState", "DialogueStateStore"]
