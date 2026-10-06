# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单05 - Query 理解优化（多轮对话 + 指代消解/省略补全）
"""
会话记忆：服务端按 session_id 保存轮次，供多轮对话的指代消解使用。

【为什么放服务端而不是让前端回传 history】
草单里 `ChatRequest.history` 是个从未被引用的死字段。若改成"前端把历史带上来"：
  · 刷新页面即丢上下文；
  · 「会话隔离」没有可演示的载体（前端想串就串）；
  · 更关键 —— **指代消解需要的不是对话文本，而是结构化状态**：会话焦点实体
    （「他」指谁）与问点模板（「那力源呢？」继承的是哪个问法）。这两样从
    `[{role,content}]` 里反解既脆弱又不可靠。
所以焦点实体/问点模板由服务端在每轮结束后写回会话，多轮只靠一个 id。

【为什么进程内、不引入 Redis】
本工单的主题是「指代消解 / 省略补全」，与会话存储介质无关。引入 Redis 只是把
「进程重启即失效」换成「redis 重启即失效」，换来一个新依赖与一套新的失效模式。
所以刻意用进程内存储，并把这个限度**写进文档**，不藏。

【线程安全】`api/evaluate` 走 `asyncio.to_thread`，所以这里必须加锁；
且**锁内绝不 await**（一旦 await 就有可能在持锁期间被切走）。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from app.config import settings

# 问点模板里的公司槽位。用一个原文绝不会出现的符号，避免与正文撞车。
COMPANY_SLOT = "⟦公司⟧"


@dataclass
class Turn:
    """一轮问答。"""

    question: str                    # 用户原话（展示用）
    rewritten: str                   # 改写后的独立问题（检索/生成/history 都用它）
    answer: str = ""
    # 本轮结束后的会话焦点实体（**规范全称**）。下一轮的「他/这个公司」指向它。
    focus_entity: str = ""
    # 本轮的问点模板，形如「⟦公司⟧法定代表人是谁？」。下一轮「那X呢？」代入 X。
    intent_template: str = ""
    doc_key: str = ""                # 本轮路由到的文档 key（排查用）
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    turns: deque[Turn]
    created: float
    last_active: float


@dataclass
class SessionLookup:
    """`get_or_create` 的结果。

    `reason` 必须能区分三种「新建」—— 验收③要演示 TTL 过期，只报"新建了会话"
    是拍不出证据的：
      · "new"     首次见到的 id
      · "expired" 见过、但已超 TTL（**这是 TTL 演示要拍的那一种**）
      · "evicted" 见过、但被 LRU 挤出去了（会话数超上限；与 TTL 不是一回事）
      · "active"  命中已有会话
    """

    session: Session
    reason: str = "active"

    @property
    def recreated(self) -> bool:
        return self.reason in ("new", "expired", "evicted")

    @property
    def expired(self) -> bool:
        return self.reason == "expired"


class SessionStore:
    def __init__(self, *, ttl: float | None = None, max_turns: int | None = None,
                 max_sessions: int | None = None) -> None:
        self.ttl = settings.session_ttl_seconds if ttl is None else ttl
        self.max_turns = settings.session_max_turns if max_turns is None else max_turns
        self.max_sessions = (settings.session_max_sessions if max_sessions is None
                             else max_sessions)
        self._lock = threading.RLock()
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        # 见过的 id（含已过期/已逐出的）。TTL 与 LRU 的差别全靠它才能分辨。
        self._seen: "OrderedDict[str, float]" = OrderedDict()
        self._seen_max = max(256, self.max_sessions * 4)

    # ------------------------------------------------------------------
    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def _purge_locked(self, now: float) -> None:
        """惰性清理：把超 TTL 的会话摘掉（不引入后台任务，少一个生命周期）。"""
        dead = [sid for sid, s in self._sessions.items() if now - s.last_active > self.ttl]
        for sid in dead:
            del self._sessions[sid]

    def _remember_locked(self, sid: str, now: float) -> None:
        self._seen[sid] = now
        self._seen.move_to_end(sid)
        while len(self._seen) > self._seen_max:
            self._seen.popitem(last=False)

    # ------------------------------------------------------------------
    def get_or_create(self, sid: str | None) -> SessionLookup:
        now = time.time()
        with self._lock:
            if not sid:
                sid = self.new_id()
                sess = Session(id=sid, turns=deque(maxlen=self.max_turns),
                               created=now, last_active=now)
                self._sessions[sid] = sess
                self._remember_locked(sid, now)
                self._evict_locked()
                return SessionLookup(sess, "new")

            self._purge_locked(now)
            sess = self._sessions.get(sid)
            if sess is not None:
                sess.last_active = now
                self._sessions.move_to_end(sid)
                self._remember_locked(sid, now)
                return SessionLookup(sess, "active")

            reason = "expired" if sid in self._seen else "new"
            sess = Session(id=sid, turns=deque(maxlen=self.max_turns),
                           created=now, last_active=now)
            self._sessions[sid] = sess
            self._remember_locked(sid, now)
            self._evict_locked()
            return SessionLookup(sess, reason)

    def _evict_locked(self) -> None:
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    # ------------------------------------------------------------------
    def append(self, sid: str, turn: Turn) -> None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return
            sess.turns.append(turn)
            sess.last_active = turn.ts

    def get(self, sid: str) -> Session | None:
        """只读查询，**不创建、不刷新 TTL**（供 `/api/session/{id}` 用）。"""
        if not sid:
            return None
        with self._lock:
            self._purge_locked(time.time())
            return self._sessions.get(sid)

    def focus_entity(self, sid: str) -> str:
        """会话焦点实体 = 最近一轮写回的规范全称。"""
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return ""
            for t in reversed(sess.turns):
                if t.focus_entity:
                    return t.focus_entity
            return ""

    def intent_template(self, sid: str, *, lookback: int | None = None) -> str:
        """最近一个**含槽位**的问点模板，向前回溯 lookback 轮。

        【为什么必须含槽位】不含槽位的模板（如上一轮问「今年收入多少？」）
        代不进公司名，拿来拼「那力源呢？」只会把公司名无处安放。所以回溯时跳过它们。
        【为什么要回溯】用户可能先聊了一轮没有公司的泛问，再问「那力源呢？」——
        这时要拿到更早那轮的模板。
        """
        n = settings.query_intent_lookback if lookback is None else lookback
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return ""
            for t in list(reversed(sess.turns))[:max(1, n)]:
                if t.intent_template and COMPANY_SLOT in t.intent_template:
                    return t.intent_template
            return ""

    # ------------------------------------------------------------------
    def history_messages(self, sid: str, *, budget_chars: int = 0,
                         with_answers: bool | None = None,
                         max_turns: int | None = None,
                         ) -> tuple[list[dict[str, str]], bool]:
        """组装送进 prompt 的 history。返回 `(messages, dropped)`。

        【为什么 history 里放的是「改写后的问题」】原话是「他参与的哪个工程…」——
        模型看到「他」仍然不知道指谁。放改写后的独立问题，history 才真的有用。

        【为什么要有预算闸门】history 与本次 context **共享** num_ctx。
        实测 context 最大 2470 字、提示词可用约 3896 token → 余量只有约 1000 token。
        只靠 maxlen 是不够的（同样的轮数，长短问题差很多）。超预算就从**最旧的一轮**
        整轮丢弃（不能只丢 user 留 assistant —— 那会拼出没有提问的回答），
        并把 dropped 如实上报，让验收的人能亲眼看到「没超预算」。
        """
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None or not sess.turns:
                return [], False
            turns = list(sess.turns)

        if with_answers is None:
            with_answers = settings.session_history_with_answers
        if max_turns is None:
            max_turns = settings.session_history_turns
        turns = turns[-max(1, max_turns):]

        budget = settings.session_prompt_budget_chars

        def size(ts: list[Turn]) -> int:
            n = 0
            for t in ts:
                n += len(t.rewritten) + 8        # 角色/协议开销粗算
                if with_answers and t.answer:
                    n += len(t.answer) + 8
            return n

        dropped = False
        while turns and size(turns) + budget_chars > budget:
            turns.pop(0)
            dropped = True

        msgs: list[dict[str, str]] = []
        for t in turns:
            msgs.append({"role": "user", "content": t.rewritten})
            if with_answers and t.answer:
                msgs.append({"role": "assistant", "content": t.answer})
        return msgs, dropped

    def history_chars(self, sid: str, *, with_answers: bool | None = None) -> int:
        msgs, _ = self.history_messages(sid, with_answers=with_answers,
                                        max_turns=settings.session_max_turns)
        return sum(len(m["content"]) for m in msgs)

    # ------------------------------------------------------------------
    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def sweep(self) -> int:
        with self._lock:
            before = len(self._sessions)
            self._purge_locked(time.time())
            return before - len(self._sessions)

    def stats(self) -> dict:
        with self._lock:
            return {
                "n_sessions": len(self._sessions),
                "n_turns": sum(len(s.turns) for s in self._sessions.values()),
                "ttl_seconds": self.ttl,
                "max_turns": self.max_turns,
                "max_sessions": self.max_sessions,
            }


# 进程内单例。**不放进 settings** —— settings 是 lru_cache 单例且会被并发读，
# 会话状态是可变对象，混进去容易在并发下被误改（见 profiles.py 顶部同类说明）。
_STORE: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _STORE
    if _STORE is None:
        _STORE = SessionStore()
    return _STORE
