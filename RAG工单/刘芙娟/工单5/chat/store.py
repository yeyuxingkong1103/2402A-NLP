"""会话存储。**本包（乃至本特性）唯一碰 Redis 的地方。**

    create_session(user_id)              -> (session_id, SessionMeta)
    get_meta(session_id)                 -> SessionMeta
    session_exists(session_id)           -> bool
    update_user_id(session_id, user_id)  -> SessionMeta
    clear_session(session_id)            -> int
    delete_session(session_id)           -> int

    append_message(...)                  -> Message        （委托给 MessageStore）
    get_history(...)                     -> list[Message]  （委托给 MessageStore）

---

## 本模块的两个部分

**会话**（本文件）：会话的建立、存在性、元数据、生命周期。
它们的失败是"会话不存在"—— 一次判定，一个 404。

**消息**（`messages.py`）：消息 List 的读写与原子写入。
它的失败是"写入没原子"—— 高并发下才现形，表现为"少了几条"。

后者的六个方法与两个 Lua 脚本搬到 `messages.py`，直接原因是 300 行上限
（FR-037 / SC-009），实质原因是**两类失败的测试策略与改动风险评估完全不同**。
本类保留同名方法作为**门面**（facade），调用方无需知道内部如何分家。

Key 构造、Lua 脚本、错误构造在 `scripts.py`。

## 只读的部分与只写的部分

与 `backend/retrieve/store.py`（只读 Milvus）和 `backend/index/store.py`（唯一写库）
的切分同一取向：**只有本包持有 Redis 连接**，其余层通过它访问。
这让 FR-033（存储层唯一）在结构上可见，而不是靠约定。

⚠️ **本模块 MUST NOT 含业务判定**（脱敏、窗口裁剪、拒答决策都在别处）。
它只负责"把给定的字节存进去、按顺序取出来"。

「为什么用 `redis.asyncio`」与「为什么写入用 Lua 脚本」两条，随写入逻辑
一并记在 `messages.py` 的模块文档里 —— 它们回答的是**写路径**的问题，
留在这里会让读的人以为与建会话有关。
"""

from __future__ import annotations

import logging
import secrets

from . import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    DEFAULT_SESSION_TTL,
    ChatError,
)
from .codec import MessageCodec
from .messages import MessageStore
from .models import Message, SessionMeta
from .scripts import (
    FIELD_CREATED_AT,
    FIELD_USER_ID,
    FIELD_LAST_ACTIVE_AT,
    META_FIELDS,
    guest_user_id,
    now,
    atomic_append_script,
    atomic_rename_script,
    dep_error,
    meta_key,
    session_key,
    session_missing,
)

logger = logging.getLogger(__name__)

__all__ = ["ChatStore", "guest_user_id", "atomic_append_script", "atomic_rename_script"]


class ChatStore:
    """会话存储。**注入 client，便于测试替换。**

    参数全部显式传入（不读环境变量）：配置在启动期由 `serve.py` 读入并注入，
    请求路径 MUST NOT 读环境变量（FR-035）。
    """

    def __init__(
        self,
        client,
        *,
        codec: MessageCodec | None = None,
        ttl: int = DEFAULT_SESSION_TTL,
        max_history: int = DEFAULT_MAX_HISTORY_MESSAGES,
    ) -> None:
        self._client = client
        # ⚠️ `_ttl` 两边都要留一份：建会话时（本类）要设 TTL，
        #    追加消息时（`MessageStore`）要刷新它。
        #
        #    实现期实测到的缺陷：搬走消息方法时把 `_ttl` 一并搬走了，
        #    而 `create_session` 仍在读 `self._ttl` —— 抛出的 `AttributeError`
        #    被下面那个 `except Exception` 包成了 `ChatError("创建会话失败：AttributeError…")`。
        #    一条编程错误伪装成了一次外部依赖故障，报错信息里一个字都没提到
        #    "少了属性"。
        self._ttl = ttl
        # 消息相关的能力组合成一个 MessageStore 而不是继承。
        # 组合（而非继承）的理由：这两组方法之间**没有 is-a 关系**，
        # 继承会让"会话"凭空获得"消息"的全部接口，也让测试替身难以只替换一半。
        self._messages = MessageStore(
            client, codec=codec, ttl=ttl, max_history=max_history
        )

    # ------------------------------------------------------------ Key

    @staticmethod
    def session_key(session_id: str) -> str:
        return session_key(session_id)

    @staticmethod
    def meta_key(session_id: str) -> str:
        return meta_key(session_id)

    # ------------------------------------------------------------ 消息（门面）

    async def append_message(
        self, session_id: str, role: str, content: str, message_id: str
    ) -> Message:
        """追加一条消息。**实现见 `messages.MessageStore.append_message`。**"""

        return await self._messages.append_message(session_id, role, content, message_id)

    async def get_history(self, session_id: str, limit: int | None = None) -> list[Message]:
        """读历史。**实现见 `messages.MessageStore.get_history`。**"""

        return await self._messages.get_history(session_id, limit)

    # ------------------------------------------------------------ 写

    async def create_session(self, user_id: str | None = None) -> tuple[str, SessionMeta]:
        """建会话。返回 `(session_id, meta)`。

        ⚠️ **只写元数据 Key，不写消息 Key。** 空的 List 在 Redis 里不存在，
        为一个不存在的空列表专门建一条 Key 是无意义的写入 ——
        而它会立刻带来一个新问题："元数据在、消息 Key 也在但是空的"
        和"元数据在、消息 Key 不在"要不要区分？答案是它们本来就该是同一件事。
        首次 `append_message` 时才创建 List。

        `user_id` 为 `None` 或空白时自动生成（Q5）。空白也视为"没提供" ——
        一个长度为 0 的显示名与没有名字是同一件事，而让它落库只会
        让 `GET` 回来的界面显示一片空白。
        """

        session_id = secrets.token_urlsafe(32)
        resolved = (user_id or "").strip() or guest_user_id()

        ts = now()
        meta = SessionMeta(user_id=resolved, created_at=ts, last_active_at=ts)
        key = meta_key(session_id)

        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.hset(
                    key,
                    mapping={
                        FIELD_USER_ID: meta.user_id,
                        FIELD_CREATED_AT: meta.created_at,
                        FIELD_LAST_ACTIVE_AT: meta.last_active_at,
                    },
                )
                pipe.expire(key, self._ttl)
                await pipe.execute()
        except Exception as exc:  # noqa: BLE001 —— 连接层失败一律归为外部依赖
            raise dep_error(exc, "创建会话") from exc

        logger.info("chat_session_created session_id=%s", session_id)
        return session_id, meta

    async def update_user_id(self, session_id: str, user_id: str) -> SessionMeta:
        """改名。**只动 `user_id` 一个字段。**

        ⚠️ **MUST NOT 刷新 TTL，MUST NOT 更新 `last_active_at`**（Q5 裁决 A）。

        TTL 的语义是"这场对话还在继续"，改名是改设置，不是继续对话。
        若这里刷新，规则就变成"任何操作都算活跃" —— 而那条规则无法解释
        为什么读历史不算（`get_history` 同样不刷新）。

        ⚠️ **MUST NOT 顺手创建会话。** 走 Lua 脚本而不是
        `pipeline(transaction=True)` + `EXISTS`：后者的 `EXISTS` 只是**返回值**，
        拦不住后面的 `HSET`，对一个不存在的 Key 执行 `HSET` 会把它创建出来
        （实现期实测到的真实缺陷，见 `scripts.atomic_rename_script`）。
        """

        try:
            updated = await self._client.eval(
                atomic_rename_script, 1, meta_key(session_id), FIELD_USER_ID, user_id
            )
        except Exception as exc:  # noqa: BLE001
            raise dep_error(exc, "修改发起者标识") from exc

        if not updated:
            raise session_missing(session_id)

        logger.info("chat_session_renamed session_id=%s", session_id)
        # 回读一次而不是就地拼一个 `SessionMeta`：`created_at` 与
        # `last_active_at` 的真实值在 Redis 里，本地拼要假设"它们没变" ——
        # 而那个假设正是改名操作要保证的东西，不能用来构造它的返回值。
        return await self.get_meta(session_id)

    async def clear_session(self, session_id: str) -> int:
        """清空消息，**保留会话**。返回被清掉的条数。

        ⚠️ 与 `delete_session` 的差别是语义级的（FR-004）：清空后会话仍可追加，
        只是历史归零。若把两者实现成同一个操作，"清空"就等于删除。
        """

        key = session_key(session_id)
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.exists(meta_key(session_id))
                pipe.llen(key)
                pipe.delete(key)
                exists, removed, _ = await pipe.execute()
        except Exception as exc:  # noqa: BLE001
            raise dep_error(exc, "清空会话") from exc

        if not exists:
            raise session_missing(session_id)
        return int(removed)

    async def delete_session(self, session_id: str) -> int:
        """删除**两条 Key**。返回被删掉的 Key 数（0 表示本来就不存在）。

        ⚠️ **两条一起删。** 只删一条会让另一条成为无法访问的残留，
        一直占着内存到 TTL 到期 —— 与数据最小化原则冲突（FR-002）。

        ⚠️ **对不存在的会话不抛错**（与 `append_message` 不同）——
        `DELETE` 是幂等语义，"删掉一个本来就没有的东西"达成了调用方意图。
        抛错会让客户端的重试收到一个它无法处理的失败。
        """

        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.delete(session_key(session_id))
                pipe.delete(meta_key(session_id))
                # ⚠️ 两个 `delete` 各自返回**自己**删掉的 Key 数，必须相加。
                # 实现期实测到的缺陷：只取第一个结果会漏算元数据那条，
                # 于是删掉一个完整会话返回 1 而不是 2 —— 而 2 正是
                # "两条 Key 都清掉了"的唯一可断言证据。
                session_deleted, meta_deleted = await pipe.execute()
        except Exception as exc:  # noqa: BLE001
            raise dep_error(exc, "删除会话") from exc

        removed = int(session_deleted) + int(meta_deleted)
        logger.info("chat_session_deleted session_id=%s removed=%d", session_id, removed)
        return removed

    # ------------------------------------------------------------ 读

    async def get_meta(self, session_id: str) -> SessionMeta:
        """读元数据。**会话是否存在以此为准。**

        不存在（Key 没了 / TTL 到期 / 已被删除）即抛 `ChatError(session_missing=True)`。
        路由层据此返回 404，而不是把它混进 503（"稍后重试"）。
        """

        try:
            data = await self._client.hgetall(meta_key(session_id))
        except Exception as exc:  # noqa: BLE001
            raise dep_error(exc, "读取会话元数据") from exc

        if not data:
            raise session_missing(session_id)

        missing = [name for name in META_FIELDS if name not in data]
        if missing:
            # 元数据缺字段意味着它在某个时刻被写坏或被手工改过。
            # 不兜底成默认值：一个 `created_at` 为 0 的会话看起来是"1970 年建的"，
            # 而真相是"这条记录不完整" —— 两者该走的处置完全不同。
            #
            # ⚠️ 这不是"会话不存在"（`session_missing=False`）——
            #    路由层据此返回 503 而不是 404。一条损坏的记录不该让用户
            #    以为自己开错了会话。
            raise ChatError(
                "会话元数据不完整（session_id=%s，缺 %s）"
                % (session_id, "、".join(missing))
            )

        return SessionMeta(
            user_id=data[FIELD_USER_ID],
            created_at=int(data[FIELD_CREATED_AT]),
            last_active_at=int(data[FIELD_LAST_ACTIVE_AT]),
        )

    async def session_exists(self, session_id: str) -> bool:
        """会话是否还在。**不抛错**，只回布尔。

        用途是 `dialogue` 在流内判定"会话还在不在"（contracts/chat-sse.md §4）——
        那里需要的是一个是非答案，而 `get_meta` 的异常语义在这里反而碍事。
        """

        try:
            return bool(await self._client.exists(meta_key(session_id)))
        except Exception as exc:  # noqa: BLE001
            raise dep_error(exc, "判定会话是否存在") from exc
