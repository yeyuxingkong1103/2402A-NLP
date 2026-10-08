"""上下文窗口裁剪。**存储历史 → 模型输入的唯一转换点。**

    build_context_window(history, *, system_prompt, max_tokens) -> list[Message]

---

## 为什么不能"只取最近 N 条"

多轮对话的核心场景是**追问**，而追问省略了主语（"那饮食上呢？"）。
若窗口把首条主诉裁掉，模型就失去了判断"他在问什么"的唯一线索 ——
它只能猜，或者答非所问。

保留主诉是让追问能工作的**结构性前提**，不是优化。

因此填充顺序是（FR-010）：

    ① 系统角色设定      —— 无条件保留
    ② 最早的用户主诉     —— 无条件保留
    ③ 最近的若干轮       —— 用剩余预算从最新往回填

## 与 TTL 的关系：两套独立的截断机制

| | TTL / LTRIM（`store.py`） | 本模块的 token 预算 |
|---|---|---|
| 管什么 | **存多少** | **送多少给模型** |
| 什么时候生效 | 每次写入 / 超时 | 每次读上下文 |

把它们混为一谈的典型后果是"以为 TTL 会清理，所以不必限长"（R4）。
本模块 MUST NOT 反过来去动存储 —— 它只读，不写。
"""

from __future__ import annotations

import logging

from . import ROLE_SYSTEM, ROLE_USER
from .models import Message
from .tokens import estimate_tokens

logger = logging.getLogger(__name__)

__all__ = ["build_context_window", "SYSTEM_MESSAGE_ID"]

# 系统角色设定的合成标识。
#
# ⚠️ 角色设定**不存 Redis**（它是进程级常量，每条会话都一样，存进去是白占空间），
# 因此这里为它合成一个 `Message`。它的 `message_id` / `timestamp` 永远
# 不会离开本进程 —— `Message.as_chat_message()` 只输出 role 与 content。
SYSTEM_MESSAGE_ID = "system-prompt"


def _system_message(system_prompt: str) -> Message:
    return Message(
        role=ROLE_SYSTEM,
        content=system_prompt,
        timestamp=0,
        message_id=SYSTEM_MESSAGE_ID,
    )


def build_context_window(
    history: list[Message],
    *,
    system_prompt: str,
    max_tokens: int,
) -> list[Message]:
    """按优先级裁出送入模型的上下文，返回**按时间正序**的列表。

    边界情形全部有确定行为，且**都留日志**（FR-013）—— 静默截断会让
    "为什么模型忽然不记得前面说的话"变成一个无从排查的现象。

    | 情形 | 行为 | 日志 |
    |---|---|---|
    | 角色设定本身超预算 | 只返回角色设定 | WARNING |
    | 角色设定 + 主诉合计超预算 | 只返回这两条 | WARNING |
    | 主诉与"最近一轮"是同一条 | 只计一次 | 不记（正常） |
    | 预算充裕，装下全部 | 返回全部 | 不记（正常） |
    """

    system_message = _system_message(system_prompt)
    system_tokens = estimate_tokens(system_prompt)

    if system_tokens > max_tokens:
        logger.warning(
            "chat_context_system_over_budget system_tokens=%d max_tokens=%d —— "
            "只返回角色设定，本轮没有任何历史进入模型",
            system_tokens,
            max_tokens,
        )
        return [system_message]

    remaining = max_tokens - system_tokens

    # ---- ② 最早的用户主诉 ----
    first_user = next((m for m in history if m.role == ROLE_USER), None)
    if first_user is not None:
        cost = estimate_tokens(first_user.content)
        if cost > remaining:
            logger.warning(
                "chat_context_complaint_over_budget complaint_tokens=%d remaining=%d "
                "—— 只返回角色设定与首条主诉，后续轮次全部丢弃",
                cost,
                remaining,
            )
            return [system_message, first_user]
        remaining -= cost

    # ---- ③ 最近的若干轮，从最新往回填 ----
    #
    # ⚠️ **遇到第一条装不下的就停**，而不是跳过它继续往前找。
    #
    # 跳过会得到一段**不连续**的上下文：模型看到"第 1 轮、第 3 轮、第 4 轮"，
    # 而它无法知道中间缺了什么 —— 于是可能把第 3 轮的用户话当成第 1 轮的延续。
    # 保持连续（哪怕少带几轮）是更安全的选择。
    tail: list[Message] = []
    for message in reversed(history):
        if message is first_user:
            # ⚠️ 主诉已经在上面放进去了。**这里必须跳过而不是重复添加** ——
            # 只有一轮对话时（主诉 == 最近一轮）重复添加会让同一条消息
            # 在上下文里出现两次，模型会以为用户说了两遍。
            continue
        cost = estimate_tokens(message.content)
        if cost > remaining:
            break
        tail.append(message)
        remaining -= cost

    tail.reverse()
    return [system_message] + ([first_user] if first_user is not None else []) + tail
