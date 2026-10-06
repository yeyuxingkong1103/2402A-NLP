# -*- coding: utf-8 -*-
"""短期记忆与会话硬隔离。

隔离键由 User_ID + Role_ID + Session_ID 三段拼成，任何一段不同就是不同的会话，
从存储层面杜绝串号（对应设计文档里的「硬隔离机制：防止串号」）。

**B-9（r16 §20）改判**：Redis **只作上下文窗口**——最多保留**最近 5 组「问+答」**
（= 10 条消息），完整历史由关系库（SQLite `messages` 表）承载。
窗口裁剪**按"组"对齐**（不会把某一组切成半截），且**裁剪由调用方决定时机**：
``/chat`` 收尾必须**先写成功 Milvus 长期记忆、再裁剪**，所以本模块提供
:meth:`SessionStore.trim_window` 供显式调用（`append` 在窗口模式下**不再自动裁剪**，
只保留一个防失控的硬上限）。

两个实现：
  * RedisSessionStore —— 真实实现；
  * InMemorySessionStore —— Redis 缺席时的兜底，保证服务仍能起来。
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

from ..config import RagConfig
from ..schemas import Message
from ..utils import stable_id

logger = logging.getLogger(__name__)

SEP = "\x1f"          # 不可见分隔符，避免 user/role/session 拼接歧义
DEFAULT_PREFIX = "legal_rag:session:"
#: 上下文窗口 = 最近 5 组「问+答」（r16 §20.2 定稿：1 组 = user + assistant = 2 条）
DEFAULT_WINDOW_TURNS = 5
#: 防失控硬上限（组数的倍数）：只有在"长期记忆写失败 → 暂不裁剪"时才会越过 5 组；
#: 超过它必须**显式告警**再裁，绝不静默增长（`AC-ST-9` 的反面）。
HARD_LIMIT_TURN_FACTOR = 8


def window_turns_from_env(default: int = DEFAULT_WINDOW_TURNS) -> int:
    """窗口组数（``SESSION_WINDOW_TURNS``，默认 5；非法值回落默认）。"""
    try:
        value = int(str(os.environ.get("SESSION_WINDOW_TURNS", "")).strip() or default)
    except ValueError:
        return default
    return value if value >= 1 else default


def session_key(user_id: str, role_id: str, session_id: str,
                prefix: str = DEFAULT_PREFIX) -> str:
    """会话键（**含用户维度**，服务端从登录态派生；`AC-CI-7`）。"""
    return f"{prefix}{user_id}{SEP}{role_id}{SEP}{session_id}"


def message_id_for(user_id: str, session_id: str, role: str, content: str, *,
                   seq: int | None = None) -> str:
    """消息的稳定 ID。

    两种口径（**给了 ``seq`` 就用位置寻址**）：

    * ``seq`` 给定 → **位置寻址**：身份 = ``(user_id, session_id, role, seq)``，
      其中 ``seq`` 是这条消息在**该会话内的序号**（关系库的 ``messages.seq``）。
    * ``seq`` 为 ``None`` → **内容寻址**（旧口径，向后兼容）：
      身份 = ``(user_id, session_id, role, content)``。

    **为什么要位置寻址**（真机实测，2026-09-26，见 `handoff/B-9-VERIFICATION.md` §8）：
    内容寻址把"同一段文字"当成了"同一条消息"，于是同一会话里重问同一个问题时，

    * **用户消息**逐字相同 ⇒ 算出同一个 ID ⇒ 落库被唯一约束跳过，
      而**助手消息**的正文是生成出来的、逐字会变 ⇒ 照常插入
      ⇒ 关系库留下**没有提问的孤儿助手行**（会话历史里表现为"凭空的回答"）；
    * 反过来也成立：不同问题若得到**逐字相同**的回答（拒答模板就会），
      ⇒ 助手消息被跳过而用户消息插入 ⇒ 留下**有问无答**的半截轮。

    位置寻址让"重问"变成**新的一轮**（有问有答、两条都在），
    同时"同一轮重复投递"（同一个 ``seq``）仍然算出同一个 ID ⇒ 幂等不变（`AC-ST-8`）。
    """
    if seq is not None:
        return stable_id("msg", str(user_id), str(session_id), str(role), int(seq))
    return stable_id("msg", str(user_id), str(session_id), str(role), str(content))


def question_key(user_id: str, session_id: str, content: str) -> str:
    """一轮问答的**问题内容键**（长期记忆去重用，与消息 ID 口径**解耦**）。

    为什么单独一个键、不直接复用 ``message_id``：消息 ID 的身份口径会变
    （内容寻址 ↔ 位置寻址，见 :func:`message_id_for`），而"**是不是同一个问题**"
    只由问题正文决定 —— 长期记忆的"同一问题不重复沉淀"应该跟着**问题**走，
    不该跟着 ID 口径走（否则改一次 ID 口径就把去重悄悄改坏）。
    """
    return stable_id("q", str(user_id), str(session_id), str(content))


def with_pair_ids(messages: list[dict], *, user_id: str = "",
                  session_id: str = "") -> list[dict]:
    """给消息补上 ``pair_id``（一轮 = 一条 ``user`` 及其后紧跟的 ``assistant``）。

    ``pair_id`` = **该轮问题的内容键**（:func:`question_key`），助手记录在长期记忆里的
    主键取自它 ⇒ **同一个问题重问只 upsert、不追加**（详见 `B-9-VERIFICATION.md` §8）。

    只在**配对位置**上取 user→assistant：窗口是按"组"对齐的（见 :func:`split_window`），
    所以相邻即同轮。**不修改**原列表（返回浅拷贝），也不改 ``message_id`` 本身。
    """
    out: list[dict] = []
    current = ""
    for message in messages or []:
        item = dict(message)
        role = str(item.get("role") or "")
        if role == "user":
            current = question_key(user_id, session_id, str(item.get("content") or ""))
            item["pair_id"] = current
        elif role == "assistant":
            item["pair_id"] = current
        out.append(item)
    return out


def split_window(messages: list[dict], turns: int = DEFAULT_WINDOW_TURNS
                 ) -> tuple[list[dict], list[dict]]:
    """把消息序列切成 ``(窗口内保留, 滚出窗口)``，**按"组"对齐**。

    一组 = 一条 ``role=user`` 及其后的 ``role=assistant``；保留最近 ``turns`` 组的
    **起点**之后的所有消息（因此末组只落了 user 时，窗口是 4 组 + 1 条 = 9 条，
    不会出现"以 assistant 开头"的半截组）。
    """
    keep_turns = max(int(turns), 1)
    seen_users = 0
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        if str(messages[index].get("role")) == "user":
            seen_users += 1
            if seen_users == keep_turns:
                start = index
                break
    if seen_users < keep_turns:
        return list(messages), []          # 还不够 5 组：全留
    return list(messages[start:]), list(messages[:start])


class SessionStore(ABC):
    name = "base"

    def __init__(self, window: int = 20, *, turns: int | None = None) -> None:
        self.window = max(int(window), 1)
        #: ``None`` = 旧口径（按**条数**裁剪，向后兼容既有调用方）；
        #: 给值 = B-9 口径（按**组**裁剪，最多 ``2*turns`` 条）
        self.turns = int(turns) if turns else None

    # ---------- 口径 ----------
    @property
    def message_window(self) -> int:
        """窗口最多保留**多少条**消息。"""
        return (2 * int(self.turns)) if self.turns else self.window

    def hard_limit(self) -> int:
        """防失控硬上限（条数）。"""
        if not self.turns:
            return self.window
        return int(HARD_LIMIT_TURN_FACTOR) * 2 * int(self.turns)

    @abstractmethod
    def history(self, user_id: str, role_id: str, session_id: str) -> list[Message]:
        ...

    @abstractmethod
    def raw_messages(self, user_id: str, role_id: str, session_id: str) -> list[dict]:
        """**未做窗口切片前的全部缓冲**（带 ``message_id``），用于算"哪些组滚出窗口"。"""
        ...

    @abstractmethod
    def append(self, user_id: str, role_id: str, session_id: str, message: Message,
               *, seq: int | None = None) -> None:
        """追加一条消息。``seq`` 给定时按**位置寻址**取 ID（见 :func:`message_id_for`）。"""
        ...

    @abstractmethod
    def trim_window(self, user_id: str, role_id: str, session_id: str) -> int:
        """按窗口口径裁剪；返回**裁剪后**剩余条数（组模式 = 按组对齐）。"""
        ...

    @abstractmethod
    def clear(self, user_id: str, role_id: str, session_id: str) -> int:
        ...

    @abstractmethod
    def list_sessions(self, user_id: str) -> list[str]:
        ...

    def prompt_history(self, user_id: str, role_id: str, session_id: str) -> list[Message]:
        """**送入模型**的上下文（与"窗口里存了什么"不是一回事）。

        窗口口径 = 最近 5 组问答；本轮提问由引擎拼在上下文末尾，因此这里只给
        **最近 4 组已完成问答** ⇒ 提示词里恰好是"最近 5 组（含当前这一问）"，
        第 6 组（更早的）**不可能**进入提示词（`AC-ST-2`/`AC-ST-3`）。
        """
        if not self.turns:
            return self.history(user_id, role_id, session_id)
        keep, _ = split_window(self.raw_messages(user_id, role_id, session_id),
                               max(int(self.turns) - 1, 1))
        return [Message(role=str(item.get("role") or "user"),
                        content=str(item.get("content") or ""),
                        created_at=float(item.get("created_at") or 0.0),
                        citations=list(item.get("citations") or []))
                for item in keep]

    def append_turn(self, user_id: str, role_id: str, session_id: str,
                    question: str, answer: str,
                    citations: list[dict] | None = None, *,
                    seq: int | None = None) -> list[str]:
        """写一轮问答（user + assistant），返回这轮两条消息的 ``message_id``。

        ``citations`` 是**助手那条消息**当时的引用来源（用户裁决：历史要保留依据，
        刷新后仍能复核）。默认 ``None`` = 无引用来源，旧调用方一行都不用改。

        ``seq`` 是**该轮用户消息在会话内的序号**（关系库的 ``messages.seq``）。
        给了它，两条消息就按**位置**取 ID，与关系库**逐字一致** ——
        这是 `_sync_window_into_db` 能对上账的前提（否则回填会重复插入）。
        不给则退回内容寻址（老调用方 / 没有关系库的兜底路径）。
        """
        user_id_text = str(user_id)
        user_message = Message(role="user", content=question)
        assistant_message = Message(role="assistant", content=answer,
                                   citations=list(citations or []))
        assistant_seq = None if seq is None else int(seq) + 1
        self.append(user_id, role_id, session_id, user_message, seq=seq)
        self.append(user_id, role_id, session_id, assistant_message, seq=assistant_seq)
        return [message_id_for(user_id_text, str(session_id), "user", question, seq=seq),
                message_id_for(user_id_text, str(session_id), "assistant", answer,
                               seq=assistant_seq)]

    @staticmethod
    def _dump_dict(item: dict) -> str:
        return json.dumps({k: item.get(k) for k in
                           ("message_id", "role", "content", "created_at", "citations")},
                          ensure_ascii=False)

    @staticmethod
    def _load_dict(raw: str) -> dict:
        data = json.loads(raw)
        role = str(data.get("role") or "user")
        content = str(data.get("content") or "")
        return {"message_id": str(data.get("message_id") or ""),
                "role": role, "content": content,
                "created_at": float(data.get("created_at") or 0.0),
                "citations": list(data.get("citations") or [])}

    def health(self) -> dict:
        return {"provider": self.name, "window": self.window,
                "window_turns": self.turns,
                "message_window": self.message_window,
                "prefix": DEFAULT_PREFIX}


class InMemorySessionStore(SessionStore):
    name = "memory"

    def __init__(self, window: int = 20, *, turns: int | None = None) -> None:
        super().__init__(window, turns=turns)
        self._data: dict[str, list[dict]] = {}

    def _bucket(self, user_id: str, role_id: str, session_id: str) -> list[dict]:
        return self._data.get(session_key(user_id, role_id, session_id), [])

    def raw_messages(self, user_id: str, role_id: str, session_id: str) -> list[dict]:
        return [dict(item) for item in self._bucket(user_id, role_id, session_id)]

    def history(self, user_id: str, role_id: str, session_id: str) -> list[Message]:
        rows = self._bucket(user_id, role_id, session_id)[-self.message_window:]
        return [Message(role=r["role"], content=r["content"],
                        created_at=r["created_at"], citations=list(r["citations"]))
                for r in rows]

    def append(self, user_id: str, role_id: str, session_id: str, message: Message,
               *, seq: int | None = None) -> None:
        key = session_key(user_id, role_id, session_id)
        bucket = self._data.setdefault(key, [])
        bucket.append({"message_id": message_id_for(str(user_id), str(session_id),
                                                    message.role, message.content, seq=seq),
                       "role": message.role, "content": message.content,
                       "created_at": float(message.created_at or 0.0),
                       "citations": list(getattr(message, "citations", None) or [])})
        self._enforce(key, bucket)

    def _enforce(self, key: str, bucket: list[dict]) -> None:
        if self.turns:
            if len(bucket) > self.hard_limit():
                logger.warning("会话 %s 缓冲 %d 条超过硬上限 %d，强制裁剪（不静默）",
                               key, len(bucket), self.hard_limit())
                keep, _ = split_window(bucket, self.hard_limit() // 2)
                bucket[:] = keep
            return                                    # 正常裁剪交给 trim_window
        if len(bucket) > self.window:
            del bucket[: len(bucket) - self.window]

    def trim_window(self, user_id: str, role_id: str, session_id: str) -> int:
        key = session_key(user_id, role_id, session_id)
        bucket = self._data.get(key, [])
        if self.turns and bucket:
            keep, _ = split_window(bucket, self.turns)
            bucket[:] = keep
        return len(bucket)

    def clear(self, user_id: str, role_id: str, session_id: str) -> int:
        key = session_key(user_id, role_id, session_id)
        removed = len(self._data.pop(key, []))
        logger.info("清空会话 %s（%d 条消息）", key, removed)
        return removed

    def list_sessions(self, user_id: str) -> list[str]:
        marker = f"{DEFAULT_PREFIX}{user_id}{SEP}"
        result: list[str] = []
        for key in self._data:
            if key.startswith(marker):
                rest = key[len(marker):]
                parts = rest.split(SEP)
                if len(parts) >= 2:
                    result.append(f"{parts[0]}{SEP}{parts[1]}")
        return sorted(result)


class RedisSessionStore(SessionStore):
    name = "redis"

    def __init__(self, url: str = "redis://127.0.0.1:6379/0", window: int = 20,
                 ttl_seconds: int = 7 * 24 * 3600, prefix: str = DEFAULT_PREFIX,
                 *, turns: int | None = None) -> None:
        super().__init__(window, turns=turns)
        import redis  # type: ignore

        self.prefix = prefix
        self.ttl = ttl_seconds
        self._client = redis.from_url(url, decode_responses=True)
        self._client.ping()   # 连不上就直接抛，由工厂函数决定降级

    # ---------- 内部 ----------
    def _key(self, user_id: str, role_id: str, session_id: str) -> str:
        return session_key(user_id, role_id, session_id, self.prefix)

    def raw_messages(self, user_id: str, role_id: str, session_id: str) -> list[dict]:
        raw = self._client.lrange(self._key(user_id, role_id, session_id), 0, -1)
        return [self._load_dict(item) for item in raw]

    # ---------- SessionStore 接口 ----------
    def history(self, user_id: str, role_id: str, session_id: str) -> list[Message]:
        raw = self._client.lrange(self._key(user_id, role_id, session_id),
                                  -self.message_window, -1)
        messages = []
        for item in raw:
            data = self._load_dict(item)
            messages.append(Message(role=data["role"], content=data["content"],
                                    created_at=data["created_at"],
                                    citations=list(data["citations"])))
        return messages

    def append(self, user_id: str, role_id: str, session_id: str, message: Message,
               *, seq: int | None = None) -> None:
        key = self._key(user_id, role_id, session_id)
        payload = self._dump_dict({
            "message_id": message_id_for(str(user_id), str(session_id),
                                         message.role, message.content, seq=seq),
            "role": message.role, "content": message.content,
            "created_at": float(message.created_at or 0.0),
            "citations": list(getattr(message, "citations", None) or [])})
        pipe = self._client.pipeline()
        pipe.rpush(key, payload)
        pipe.expire(key, self.ttl)
        if not self.turns:
            pipe.ltrim(key, -self.window, -1)     # 旧口径：按条数滑动窗口
        pipe.execute()
        if self.turns:
            self._enforce_hard_limit(key)

    def _enforce_hard_limit(self, key: str) -> None:
        """只在**超过硬上限**时兜底裁剪（正常裁剪由 ``trim_window`` 显式调用）。"""
        length = int(self._client.llen(key))
        if length <= self.hard_limit():
            return
        logger.warning("会话 %s 缓冲 %d 条超过硬上限 %d，强制按窗口裁剪（不静默）",
                       key, length, self.hard_limit())
        self._trim_key(key)

    def _trim_key(self, key: str) -> int:
        raw = self._client.lrange(key, 0, -1)
        if not raw:
            return 0
        if not self.turns:
            self._client.ltrim(key, -self.window, -1)
            return int(self._client.llen(key))
        keep, _ = split_window([self._load_dict(item) for item in raw], self.turns)
        if len(keep) < len(raw):
            self._client.ltrim(key, len(raw) - len(keep), -1)
        return int(self._client.llen(key))

    def trim_window(self, user_id: str, role_id: str, session_id: str) -> int:
        return self._trim_key(self._key(user_id, role_id, session_id))

    def clear(self, user_id: str, role_id: str, session_id: str) -> int:
        key = self._key(user_id, role_id, session_id)
        length = int(self._client.llen(key))
        self._client.delete(key)
        logger.info("清空会话 %s（%d 条消息）", key, length)
        return length

    def list_sessions(self, user_id: str) -> list[str]:
        pattern = f"{self.prefix}{user_id}{SEP}*"
        result: list[str] = []
        for key in self._client.scan_iter(match=pattern, count=100):
            rest = key[len(f"{self.prefix}{user_id}{SEP}"):]
            parts = rest.split(SEP)
            if len(parts) >= 2:
                result.append(f"{parts[0]}{SEP}{parts[1]}")
        return sorted(result)

    def count_own_keys(self, limit: int = 1000) -> int:
        """只数**自己前缀**下的 key（用 SCAN，绝不 ``KEYS *``、绝不 flush）。

        用来证明「服务没碰用户既有 key」：这个数字只应随本服务的会话增长。
        """
        total = 0
        for _ in self._client.scan_iter(match=f"{self.prefix}*", count=200):
            total += 1
            if total >= limit:
                break
        return total

    def count_user_keys(self, user_id: str, limit: int = 1000) -> int:
        """只数**某个用户维度**下的 key（`AC-CI-7` 的 key 前缀计数证据）。"""
        total = 0
        for _ in self._client.scan_iter(match=f"{self.prefix}{user_id}{SEP}*", count=200):
            total += 1
            if total >= limit:
                break
        return total

    def health(self) -> dict:
        info = dict(super().health())
        info["prefix"] = self.prefix
        info["ttl"] = self.ttl
        try:
            info["ping"] = bool(self._client.ping())
            # 只读统计；失败不影响健康检查结论
            info["own_keys"] = self.count_own_keys()
        except Exception as exc:  # noqa: BLE001
            info["ping"] = False
            info["error"] = str(exc)
        return info


def build_session_store(config: RagConfig) -> SessionStore:
    """优先 Redis；连不上就降级到内存实现并打印清晰日志。

    **B-9 口径**：窗口 = 最近 ``SESSION_WINDOW_TURNS``（默认 5）组「问+答」，
    按"组"对齐裁剪（显式 ``trim_window``）。
    """
    turns = window_turns_from_env()
    window = 2 * turns
    try:
        store = RedisSessionStore(config.redis_url, window, turns=turns)
        logger.info("短期记忆使用 Redis: %s（上下文窗口=最近 %d 组问答 = %d 条）",
                    config.redis_url, turns, window)
        return store
    except Exception as exc:  # noqa: BLE001 - 降级是预期路径
        logger.warning("Redis 不可用（%s），短期记忆降级为内存实现", exc)
        return InMemorySessionStore(window, turns=turns)


__all__ = ["SessionStore", "InMemorySessionStore", "RedisSessionStore",
           "build_session_store", "session_key", "message_id_for", "split_window",
           "with_pair_ids", "question_key", "DEFAULT_WINDOW_TURNS", "window_turns_from_env"]
