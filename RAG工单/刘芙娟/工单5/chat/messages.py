"""消息 List 的读写。**原子写入的唯一实现处。**

    MessageStore.append_message(session_id, role, content, message_id) -> Message
    MessageStore.get_history(session_id, limit)                       -> list[Message]

---

## 与 `store.py` 的分界

`store.py` 管**会话的存在与元数据**（建、查、改名、清空、删除），
本模块管**消息 List 的内容**（追加、读取）。

拆开的实质理由不是行数，是**失败模式不同**：

    store.py    的失败是"会话不存在" —— 一次判定，一个 404
    messages.py 的失败是"写入没原子" —— 高并发下才现形，且表现为"少了几条"

前者靠存在性检查就能防，后者只能靠 Lua 脚本 + 并发测试。两者的测试策略、
改动风险评估、乃至"什么算改对了"都不一样，放在一个文件里会互相稀释。

## 为什么 `content` 的脱敏不在这里

本模块被两个场景调用：存用户消息、存助手消息。两者的脱敏时机不同
（助手消息在装配之后）。把脱敏放在这里会让"哪条消息脱过敏"取决于调用顺序；
放在调用方则**每个入口各自负责**，漏掉一个是显式的、可测试的。

⚠️ 不存在"两处脱敏"的问题：`redact()` 幂等，重复调用无害。

---

## 为什么用 `redis.asyncio`

`dialogue.py` 必须 `await call_model()`，因此编排层本身是协程。若这里用同步客户端，
每次 Redis 往返都会阻塞事件循环 —— 与 `backend/generate/client.py` 模块文档里
「同步客户端会把这 5–10 秒整个事件循环堵住」是同一个问题。

## 为什么写入用 Lua 脚本

「追加消息 + 截断到上限 + 刷新两条 TTL + 更新 `last_active_at`」是 **5 个操作**。
用 `WATCH` 写要变成重试循环（读 → 监视 → 改 → 提交，冲突整体重来），
带来不可控的重试次数与自己的失败模式；而本接口的延迟预算里已经有一半被生成占用。

Lua 在 Redis 服务端**单线程原子执行**，一次往返，无重试分支。详见 `research.md` R2。
"""

from __future__ import annotations

import logging

from . import DEFAULT_MAX_HISTORY_MESSAGES, DEFAULT_SESSION_TTL
from .codec import JsonCodec, MessageCodec
from .models import Message
from .scripts import (
    FIELD_LAST_ACTIVE_AT,
    now,
    atomic_append_script,
    dep_error,
    meta_key,
    session_key,
    session_missing,
)

logger = logging.getLogger(__name__)

__all__ = ["MessageStore"]


class MessageStore:
    """会话消息 List 的读写。**注入 client，便于测试替换。**"""

    def __init__(
        self,
        client,
        *,
        codec: MessageCodec | None = None,
        ttl: int = DEFAULT_SESSION_TTL,
        max_history: int = DEFAULT_MAX_HISTORY_MESSAGES,
    ) -> None:
        self._client = client
        # 注入 codec（而非内部 new 一个）：这是 FR-027 的合规扩展点 ——
        # 将来接加密存储只需换这个对象，本类的方法体一行不改。
        self._codec: MessageCodec = codec or JsonCodec()
        self._ttl = ttl
        self._max_history = max_history

    async def append_message(
        self, session_id: str, role: str, content: str, message_id: str
    ) -> Message:
        """追加一条消息，并原子地截断 + 刷新 TTL + 更新 `last_active_at`。

        ⚠️ **`role` / `content` 的合法性由调用方保证**：`role` 经
        `models.Message` 的构造校验（闭集），`content` 已脱敏。
        本方法只负责"把给定的字节原子地存进去"。

        ⚠️ **`message_id` 由调用方生成**：它可能已经用于本次请求的日志关联，
        在这里重新生成会让"日志里的 id"与"库里的 id"对不上。
        """

        message = Message(
            role=role, content=content, timestamp=now(), message_id=message_id
        )
        payload = self._codec.dumps(message)

        try:
            length = await self._client.eval(
                atomic_append_script,
                2,
                session_key(session_id),
                meta_key(session_id),
                payload,
                self._ttl,
                self._max_history,
                message.timestamp,
            )
        except Exception as exc:  # noqa: BLE001 —— 连接层失败一律归为外部依赖
            raise dep_error(exc, "追加消息") from exc

        if not length:
            # Lua 里 RPUSH 之后 LLEN 恒 ≥ 1，因此 0 只可能来自开头的
            # `EXISTS` 分支 —— 会话不存在。
            raise session_missing(session_id)

        # ⚠️ 只记长度，MUST NOT 记内容（FR-025）。
        logger.debug(
            "chat_message_appended session_id=%s message_id=%s role=%s chars=%d len=%d",
            session_id,
            message.message_id,
            role,
            len(content),
            length,
        )
        return message

    async def get_history(
        self, session_id: str, limit: int | None = None
    ) -> list[Message]:
        """按时间正序返回历史。`limit` 取**最近 N 条**（返回仍为正序）。

        ⚠️ **读 MUST NOT 刷新 TTL。** 否则一个只读不写的客户端能把会话无限续命
        —— 与数据最小化原则冲突（FR-002）。

        ⚠️ **本方法不做存在性判定。** 它只读消息 List，而"会话存在但没有消息"
        与"会话不存在"在 List 这一层**无法区分**（两者都是"Key 不存在"）。
        解析不了的记录会抛错（`codec.loads`），那是数据损坏，与不存在是两回事。
        """

        try:
            if limit is None:
                raw = await self._client.lrange(session_key(session_id), 0, -1)
            elif limit <= 0:
                # 由接口层拦（`limit > 0`），走到这里说明调用方绕过了校验。
                # 返回空而不是抛错：调用方要的是"没有消息"，这个返回值是对的。
                return []
            else:
                raw = await self._client.lrange(session_key(session_id), -limit, -1)
        except Exception as exc:  # noqa: BLE001
            raise dep_error(exc, "读取历史") from exc

        if not raw:
            return []
        return [self._codec.loads(item) for item in raw]

    async def count(self, session_id: str) -> int:
        """消息条数。`get_history` + `len` 在长历史下要反序列化全部记录，而这里不需要。"""

        try:
            return int(await self._client.llen(session_key(session_id)))
        except Exception as exc:  # noqa: BLE001
            raise dep_error(exc, "统计消息条数") from exc


# `FIELD_LAST_ACTIVE_AT` 由 Lua 脚本以字面量写入（脚本里不能 import 常量）。
# 这里导入它是为了让下面的断言在**导入期**发现两边不一致 ——
# 否则改了常量、忘了改脚本，症状是"最后活跃时间字段名对不上"，
# 而脚本不报错，只是 HSET 了一个没人读的字段。
assert FIELD_LAST_ACTIVE_AT in atomic_append_script, (
    "Lua 脚本里的 `last_active_at` 与 scripts.FIELD_LAST_ACTIVE_AT 不一致"
)
