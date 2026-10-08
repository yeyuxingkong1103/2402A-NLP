"""会话服务。**六个核心方法，外加两条横切约束（脱敏、审计）。**

    create_session(user_id)                    -> (session_id, SessionMeta)
    append_message(session_id, role, content)  -> Message
    get_history(session_id, limit)             -> list[Message]
    get_context_window(session_id, max_tokens) -> list[Message]
    clear_session(session_id)                  -> int
    delete_session(session_id)                 -> int

---

## 与 `store.py` 的分界

`store.py` 是**字节层**：给什么存什么，按顺序取回来。
`service.py` 是**语义层**：决定存之前要做什么（脱敏）、存完要记什么（审计）、
读到之后要不要裁剪（窗口）。

这条分界的实际价值是让"脱敏"和"审计"**没有第二个落点** ——
下面每个写方法都必须在同一个地方做这两件事，
而 `store.py` 里没有任何一处能绕过它们。

## 为什么 `append_message` 在这里脱敏，而不是在 `store` 里

`store.append_message` 被 `dialogue.py` 用于**两个**场景：存用户消息、存助手消息。
两者的脱敏时机不同（助手消息在装配之后、落库之前）。
把脱敏放在 `store` 里会让"哪条消息脱过敏"变得依赖调用顺序；
放在这里（以及 `dialogue` 的助手消息落库处）则**每个入口各自负责**，
漏掉一个入口是显式的、可测试的。

⚠️ 不存在"两处脱敏"的问题：`redact()` 是**幂等**的（对已脱敏文本再跑一次
不会改变结果），因此重复调用无害。
"""

from __future__ import annotations

import logging
import uuid

from . import (
    ACTION_APPEND,
    ACTION_CLEAR,
    ACTION_CREATE,
    ACTION_DELETE,
    ACTION_RENAME,
    DEFAULT_MAX_CONTEXT_TOKENS,
    ROLE_ASSISTANT,
)
from . import audit
from .context import build_context_window
from .models import Message, SessionMeta, validate_role
from .prompt import load_system_prompt
from .redact import redact
from .store import ChatStore

logger = logging.getLogger(__name__)

__all__ = ["ChatService"]


class ChatService:
    """会话的语义层。**注入 store，便于测试替换。**"""

    def __init__(self, store: ChatStore) -> None:
        self._store = store

    # ------------------------------------------------------------ 创建

    async def create_session(self, user_id: str | None = None) -> tuple[str, SessionMeta]:
        """建会话。`user_id` 缺省时由 `store` 自动生成（Q5）。

        审计记录里的 `n_messages` 传 0：新会话必然没有消息，
        而回读一次 Redis 只为拿一个已知的 0 是白花的往返。
        """

        session_id, meta = await self._store.create_session(user_id)
        audit.record(
            ACTION_CREATE, session_id, user_id=meta.user_id, n_messages=0
        )
        return session_id, meta

    # ------------------------------------------------------------ 追加

    async def append_message(self, session_id: str, role: str, content: str) -> Message:
        """追加一条消息。**脱敏在此发生。**

        `message_id` 在这里生成（而不是让 `store` 生成）：它可能已经用于
        本次请求的日志关联，在更下游生成会让"日志里的 id"与"库里的 id"对不上。

        ⚠️ **先脱敏再落库**，顺序不可颠倒 —— 落库之后再脱敏等于没有脱敏。
        """

        validate_role(role)
        safe_content = redact(content)

        message = await self._store.append_message(
            session_id, role, safe_content, str(uuid.uuid4())
        )

        audit.record(ACTION_APPEND, session_id, n_messages=None)
        return message

    # ------------------------------------------------------------ 读取

    async def get_history(self, session_id: str, limit: int | None = None) -> list[Message]:
        """读历史。**不存在的会话 → 空列表；不刷新 TTL。**

        ⚠️ 与 `append_message` 不同，这里**不抛**"会话不存在" ——
        它只读消息 List，而"会话存在但没消息"与"会话不存在"在 List 这一层
        无法区分。需要判定存在性用 `get_meta`。
        """

        await self._store.get_meta(session_id)      # 存在性判定（不存在即抛）
        return await self._store.get_history(session_id, limit)

    async def get_context_window(
        self, session_id: str, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    ) -> list[Message]:
        """裁出送入模型的上下文。**每轮都要重新调用**（FR-014）。

        ⚠️ **不得缓存或复用上一次的结果。** 每轮的检索都独立（FR-014），
        上下文同理 —— 复用会让"这一轮模型看到的历史"取决于上一轮何时发生，
        而不是取决于当前会话的真实状态。

        角色设定在此装载（`prompt.load_system_prompt`，进程内缓存），
        **不经过 Redis** —— 它是进程级常量，每条会话都一样。
        """

        history = await self._store.get_history(session_id)
        return build_context_window(
            history,
            system_prompt=load_system_prompt(),
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------ 改名

    async def rename_session(self, session_id: str, user_id: str) -> SessionMeta:
        """改名。**不刷新 TTL、不更新 `last_active_at`**（Q5 裁决 A）。"""

        meta = await self._store.update_user_id(session_id, user_id)
        audit.record(ACTION_RENAME, session_id, user_id=meta.user_id)
        return meta

    # ------------------------------------------------------------ 清空 / 删除

    async def clear_session(self, session_id: str) -> int:
        """清空消息、保留会话（FR-004）。**无对外接口**（Q3 裁决）。"""

        meta = await self._store.get_meta(session_id)
        removed = await self._store.clear_session(session_id)
        audit.record(ACTION_CLEAR, session_id, user_id=meta.user_id, n_messages=0)
        return removed

    async def delete_session(self, session_id: str) -> int:
        """删除整个会话。**幂等**，对不存在的会话返回 0 而不抛错。

        审计的 `user_id` 取不到就算了（会话已经没了）：MUST NOT 为了凑齐
        字段去回读一次 Redis —— 那不仅多一次往返，还会在"刚好 TTL 到期"
        这个竞态上抛出一个本该被吞掉的错误。
        """

        removed = await self._store.delete_session(session_id)
        audit.record(ACTION_DELETE, session_id, n_messages=None)
        return removed

    # ------------------------------------------------------------ 存在性

    async def ensure_exists(self, session_id: str) -> SessionMeta:
        """会话不存在即抛 `ChatError(session_missing=True)`；存在则返回元数据。

        用途是路由层在**流开始之前**做存在性判定（contracts/chat-sse.md §4）——
        那里需要的是"抛错 / 不抛错"，而 `get_meta` 的返回被丢弃。

        单独暴露一个方法而不是让路由去碰 `service._store`：后者会绕过本层
        直接依赖存储实现，而"只有 `store.py` 碰 Redis"这条结构约束
        是从这一层往下的，不是往上的。
        """

        return await self._store.get_meta(session_id)

    # ------------------------------------------------------------ 便捷

    async def append_exchange(
        self, session_id: str, question: str, answer_text: str
    ) -> None:
        """一次落库一问一答。**仅供 `dialogue.py` 使用。**

        ⚠️ `answer_text` MUST 是**已装配的**完整文本（含逐字话术），
        不是 `GenerationResult.answer_text`（那是模型的裸输出）。
        FR-016 要求历史里存的是"用户当时实际看到的完整文本" ——
        两者名字相同、内容不同，是本特性最容易搞错的一处。

        拆成一个方法而不是让 `dialogue` 调两次 `append_message`：
        两条消息共用同一次对话轮次，分开调用会让调用方需要自己保证
        "两条都写、且顺序正确"，而顺序错了会让历史里出现"先答后问"。
        """

        await self.append_message(session_id, "user", question)
        await self.append_message(session_id, ROLE_ASSISTANT, answer_text)
