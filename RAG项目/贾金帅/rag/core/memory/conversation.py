"""Session-isolated, bounded memory and follow-up query contextualization."""

from __future__ import annotations

import re
import threading
import time
import uuid
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable
import redis
from src.model.llm import ModelClient, get_default_model


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_FOLLOW_UP_RE = re.compile(
    r"(它|他|她|这个|那个|这款|那款|该药|此药|这种药|这个药|上述|前面|刚才|"
    r"前者|后者|第一个|第二个|其|这些|那些|剂量呢|禁忌呢|副作用呢|怎么吃|"
    r"能一起用吗|有什么风险|有什么区别|那.{0,8}呢)"
)


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    resolved_query: str
    assistant: str
    created_at: float


@dataclass(frozen=True)
class ContextualizedQuery:
    original: str
    resolved: str
    used_memory: bool


class ConversationMemoryStore:
    """Small in-process memory store with session isolation, TTL and LRU eviction."""

    def __init__(
        self,
        max_sessions: int = 500,
        max_turns: int = 6,
        ttl_seconds: int = 3600,
        max_history_chars: int = 6000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_sessions = max(1, int(max_sessions))
        self.max_turns = max(1, int(max_turns))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_history_chars = max(500, int(max_history_chars))
        self._clock = clock
        self._sessions: OrderedDict[str, list[ConversationTurn]] = OrderedDict()
        self._last_access: dict[str, float] = {}
        self._lock = threading.RLock()

    def ensure_session(self, session_id: str = "") -> str:
        candidate = (session_id or "").strip()
        if not _SESSION_ID_RE.fullmatch(candidate):
            candidate = uuid.uuid4().hex
        with self._lock:
            self._evict_expired()
            self._sessions.setdefault(candidate, [])
            self._touch(candidate)
            self._evict_overflow()
        return candidate

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        if not _SESSION_ID_RE.fullmatch((session_id or "").strip()):
            return []
        with self._lock:
            self._evict_expired()
            if session_id not in self._sessions:
                return []
            self._touch(session_id)
            return list(self._sessions[session_id])

    def append(
        self,
        session_id: str,
        user: str,
        resolved_query: str,
        assistant: str,
    ) -> None:
        if not _SESSION_ID_RE.fullmatch((session_id or "").strip()):
            return
        now = self._clock()
        turn = ConversationTurn(
            user=self._clip(user, 500),
            resolved_query=self._clip(resolved_query, 700),
            assistant=self._clip(assistant, 1800),
            created_at=now,
        )
        with self._lock:
            self._evict_expired(now)
            turns = self._sessions.setdefault(session_id, [])
            turns.append(turn)
            del turns[:-self.max_turns]
            self._touch(session_id, now)
            self._evict_overflow()

    def clear(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)
            return existed

    def format_history(self, turns: Iterable[ConversationTurn]) -> str:
        blocks: list[str] = []
        for index, turn in enumerate(turns, 1):
            blocks.append(
                f"第{index}轮\n"
                f"用户原话：{turn.user}\n"
                f"独立问题：{turn.resolved_query}\n"
                f"助手回答：{turn.assistant}"
            )
        text = "\n\n".join(blocks)
        if len(text) <= self.max_history_chars:
            return text
        return text[-self.max_history_chars :]

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        value = (value or "").strip()
        return value if len(value) <= limit else value[:limit] + "…"

    def _touch(self, session_id: str, now: float | None = None) -> None:
        self._last_access[session_id] = self._clock() if now is None else now
        self._sessions.move_to_end(session_id)

    def _evict_expired(self, now: float | None = None) -> None:
        current = self._clock() if now is None else now
        expired = [
            key
            for key, accessed_at in self._last_access.items()
            if current - accessed_at >= self.ttl_seconds
        ]
        for key in expired:
            self._sessions.pop(key, None)
            self._last_access.pop(key, None)

    def _evict_overflow(self) -> None:
        while len(self._sessions) > self.max_sessions:
            key, _ = self._sessions.popitem(last=False)
            self._last_access.pop(key, None)


class RedisConversationMemoryStore(ConversationMemoryStore):
    """Redis-backed short-term memory with local fallback when Redis is down."""

    def __init__(self, url: str, *, key_prefix: str = "medrag:memory:", **kwargs) -> None:
        super().__init__(**kwargs)
        self.key_prefix = key_prefix
        self._redis = None
        try:
            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                protocol=2,
                socket_connect_timeout=1.5,
                socket_timeout=1.5,
            )
            client.ping()
            self._redis = client
        except Exception:
            self._redis = None

    @property
    def backend_available(self) -> bool:
        return self._redis is not None

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def _load(self, session_id: str):
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._key(session_id))
            items = json.loads(raw) if raw else []
            return [ConversationTurn(**item) for item in items if isinstance(item, dict)]
        except Exception:
            self._redis = None
            return None

    def _save(self, session_id: str, turns: list[ConversationTurn]) -> bool:
        if self._redis is None:
            return False
        try:
            payload = [turn.__dict__ for turn in turns[-self.max_turns:]]
            self._redis.set(self._key(session_id), json.dumps(payload, ensure_ascii=False), ex=self.ttl_seconds)
            return True
        except Exception:
            self._redis = None
            return False

    def ensure_session(self, session_id: str = "") -> str:
        candidate = (session_id or "").strip()
        if not _SESSION_ID_RE.fullmatch(candidate):
            candidate = uuid.uuid4().hex
        loaded = self._load(candidate)
        if loaded is None and self._redis is None:
            return super().ensure_session(candidate)
        if loaded is None:
            return candidate
        self._save(candidate, loaded)
        return candidate

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        if not _SESSION_ID_RE.fullmatch((session_id or "").strip()):
            return []
        loaded = self._load(session_id)
        if loaded is None and self._redis is None:
            return super().get_history(session_id)
        if loaded:
            self._save(session_id, loaded)
        return loaded or []

    def append(self, session_id: str, user: str, resolved_query: str, assistant: str) -> None:
        if not _SESSION_ID_RE.fullmatch((session_id or "").strip()):
            return
        loaded = self._load(session_id)
        if loaded is None and self._redis is None:
            return super().append(session_id, user, resolved_query, assistant)
        turn = ConversationTurn(self._clip(user, 500), self._clip(resolved_query, 700), self._clip(assistant, 1800), time.time())
        self._save(session_id, (loaded or []) + [turn])

    def clear(self, session_id: str) -> bool:
        if self._redis is None:
            return super().clear(session_id)
        try:
            redis_cleared = bool(self._redis.delete(self._key(session_id)))
            local_cleared = super().clear(session_id)
            return redis_cleared or local_cleared
        except Exception:
            self._redis = None
            return super().clear(session_id)


class MySQLConversationMemoryStore(ConversationMemoryStore):
    """Persistent conversation memory stored in the project's configured MySQL."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ready = False
        self._db_error = ""
        try:
            from src.mysql import connect
            self._connect = connect
            self._ensure_table()
            self._ready = True
        except Exception as exc:
            self._db_error = f"{type(exc).__name__}: {exc}"

    @property
    def backend_available(self) -> bool:
        return self._ready

    def _ensure_table(self) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS conversation_memory (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    session_id VARCHAR(64) NOT NULL,
                    user_text TEXT NOT NULL,
                    resolved_query TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    created_at DOUBLE NOT NULL,
                    PRIMARY KEY (id), INDEX idx_memory_session (session_id, id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            conn.commit()
        finally:
            conn.close()

    def _fallback(self, method, *args):
        return getattr(super(), method)(*args)

    def ensure_session(self, session_id: str = "") -> str:
        candidate = (session_id or "").strip()
        if not _SESSION_ID_RE.fullmatch(candidate):
            candidate = uuid.uuid4().hex
        if not self._ready:
            return self._fallback("ensure_session", candidate)
        return candidate

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        if not _SESSION_ID_RE.fullmatch((session_id or "").strip()):
            return []
        if not self._ready:
            return self._fallback("get_history", session_id)
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute("SELECT user_text, resolved_query, assistant_text, created_at FROM conversation_memory WHERE session_id=%s ORDER BY id DESC LIMIT %s", (session_id, self.max_turns))
                rows = cur.fetchall()
            conn.close()
            return [ConversationTurn(r["user_text"], r["resolved_query"], r["assistant_text"], float(r["created_at"])) for r in reversed(rows)]
        except Exception:
            self._ready = False
            return self._fallback("get_history", session_id)

    def append(self, session_id: str, user: str, resolved_query: str, assistant: str) -> None:
        if not _SESSION_ID_RE.fullmatch((session_id or "").strip()):
            return
        if not self._ready:
            return self._fallback("append", session_id, user, resolved_query, assistant)
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute("INSERT INTO conversation_memory (session_id,user_text,resolved_query,assistant_text,created_at) VALUES (%s,%s,%s,%s,%s)", (session_id, self._clip(user, 500), self._clip(resolved_query, 700), self._clip(assistant, 1800), time.time()))
                cur.execute("SELECT id FROM conversation_memory WHERE session_id=%s ORDER BY id DESC LIMIT 18446744073709551615 OFFSET %s", (session_id, self.max_turns))
                old_ids = [row["id"] for row in cur.fetchall()]
                if old_ids:
                    cur.executemany("DELETE FROM conversation_memory WHERE id=%s", [(item,) for item in old_ids])
            conn.commit()
            conn.close()
        except Exception:
            self._ready = False
            self._fallback("append", session_id, user, resolved_query, assistant)

    def clear(self, session_id: str) -> bool:
        if not self._ready:
            return self._fallback("clear", session_id)
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_memory WHERE session_id=%s",
                    (session_id,),
                )
                changed = bool(cur.rowcount)
            conn.commit()
            return changed
        except Exception:
            if conn is not None:
                conn.rollback()
            self._ready = False
            return self._fallback("clear", session_id)
        finally:
            if conn is not None:
                conn.close()


class HybridConversationMemoryStore(ConversationMemoryStore):
    """Two-level memory: Redis hot cache backed by persistent MySQL."""

    def __init__(self, redis_url: str, *, redis_key_prefix: str = "medrag:memory:", **kwargs) -> None:
        super().__init__(**kwargs)
        self.mysql = MySQLConversationMemoryStore(**kwargs)
        self.redis = RedisConversationMemoryStore(redis_url, key_prefix=redis_key_prefix, **kwargs)

    @property
    def backend_available(self) -> bool:
        return self.mysql.backend_available or self.redis.backend_available

    def ensure_session(self, session_id: str = "") -> str:
        sid = self.mysql.ensure_session(session_id)
        return self.redis.ensure_session(sid)

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        hot = self.redis.get_history(session_id)
        if hot:
            return hot[-self.max_turns:]
        cold = self.mysql.get_history(session_id)
        if cold:
            self.redis._save(session_id, cold)
        return cold[-self.max_turns:]

    def append(self, session_id: str, user: str, resolved_query: str, assistant: str) -> None:
        self.mysql.append(session_id, user, resolved_query, assistant)
        self.redis.append(session_id, user, resolved_query, assistant)

    def clear(self, session_id: str) -> bool:
        mysql_cleared = self.mysql.clear(session_id)
        redis_cleared = self.redis.clear(session_id)
        return mysql_cleared or redis_cleared


class ConversationContextualizer:
    """Resolve pronouns and omitted context before retrieval runs."""

    SYSTEM_PROMPT = """你是检索问题改写器。你的任务不是回答问题，而是把当前追问改写成可独立检索的问题。
规则：
1. 只使用对话历史中明确出现的人物、药品、物品、疾病或比较对象来消解代词和省略。
2. 保留当前问题的真实意图、限制条件、比较顺序和否定表达，不添加任何医学事实。
3. 若代指存在歧义，不要猜测具体对象；保留原问题，或把歧义明确写入问题。
4. 历史中的助手回答可能有误，只能用于识别谈话对象，不能把其中结论加入改写结果。
5. 只输出一行改写后的问题，不要解释，不要回答，不要输出 JSON。"""

    def __init__(self, model_client: ModelClient | None = None) -> None:
        self.model_client = model_client or get_default_model()

    def contextualize(
        self, question: str, history: list[ConversationTurn]
    ) -> ContextualizedQuery:
        original = (question or "").strip()
        if not history or not self._depends_on_history(original):
            return ContextualizedQuery(original, original, False)

        history_text = self._format_for_rewrite(history)
        user_prompt = (
            f"<conversation_history>\n{history_text}\n</conversation_history>\n\n"
            f"<current_question>\n{original}\n</current_question>"
        )
        try:
            raw = self.model_client.chat(
                system=self.SYSTEM_PROMPT,
                user=user_prompt,
            )
            resolved = self._clean(raw)
        except Exception:
            resolved = ""
        if not resolved:
            anchor = history[-1].resolved_query or history[-1].user
            resolved = f"关于“{anchor}”，{original}"
        return ContextualizedQuery(original, resolved, resolved != original)

    @staticmethod
    def _depends_on_history(question: str) -> bool:
        text = (question or "").strip()
        if _FOLLOW_UP_RE.search(text):
            return True
        # Very short elliptical questions such as “用量？” or “儿童呢？”.
        return len(text) <= 12 and bool(re.search(r"(呢|吗|如何|多少|用量|风险|区别|儿童|老人)[？?]?$", text))

    @staticmethod
    def _format_for_rewrite(history: list[ConversationTurn]) -> str:
        blocks = []
        for index, turn in enumerate(history[-4:], 1):
            blocks.append(
                f"第{index}轮用户：{turn.user}\n"
                f"第{index}轮独立问题：{turn.resolved_query}\n"
                f"第{index}轮助手摘要：{turn.assistant[:600]}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _clean(raw: str) -> str:
        text = (raw or "").strip().strip("`\"'“”")
        text = re.sub(r"^(改写后的问题|独立问题|问题)\s*[：:]\s*", "", text)
        text = " ".join(text.splitlines()).strip()
        if not text or len(text) > 700:
            return ""
        return text
