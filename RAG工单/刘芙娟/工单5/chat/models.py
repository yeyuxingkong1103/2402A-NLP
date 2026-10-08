"""会话的内部数据模型。

    Message      一条消息（存储单元 + 送入模型的最小单位）
    SessionMeta  一个会话的元数据

---

## 为什么这些模型不依赖 redis

它们是**纯数据**：可以脱离存储单独构造、断言、比较。

这条约束带来的具体好处是测试：`codec.py` 的往返测试、`context.py` 的窗口裁剪测试
都不需要起一个 Redis —— 而窗口裁剪恰恰是本特性**边界最多**的一段逻辑
（系统设定超预算、主诉与最近一轮是同一条、预算刚好装下……）。
若模型带着 redis 依赖，这些边界就只能靠 fakeredis 间接验证，错误定位会绕一大圈。

## 为什么 `Message` 与 `SessionMeta` 是分开的两个类型

它们的**生命周期不同**：`clear_session` 清空消息但保留元数据（FR-004）。
合并成一个类型就无法表达"清空消息但会话还在"这件事。

这也对应 Redis 里两条 Key 的分工 —— 见 `data-model.md` §1。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER, ROLES, ChatError

__all__ = ["Message", "SessionMeta", "validate_role"]


def validate_role(role: str) -> str:
    """校验角色取值并返回它。非法即抛 `ChatError`。

    ⚠️ 角色是**闭集**（FR-007）。一个拼错的 role（`"User"`、`"human"`）
    会让模型收到一条它无法归类的消息 —— 而 Redis 里那条记录看起来完全正常，
    没有任何异常。这是典型的静默失效，因此校验放在构造点而不是信任调用方。

    MUST NOT 大小写不敏感地接受 `"User"`：OpenAI 兼容接口按字面区分角色名，
    放行一个不规范的值等于把问题推迟到一次模型调用失败。
    """

    if role not in ROLES:
        raise ChatError(
            "非法的消息角色：%r（可选：%s）" % (role, " / ".join(ROLES))
        )
    return role


@dataclass
class Message:
    """一条消息。**历史的存储单元，也是送入模型的 messages 数组的来源。**

    字段与需求原文的 JSON 结构逐字一致（role / content / timestamp / message_id）。

    可变（非 frozen）：`codec.dumps` 只读它，但让 dataclass 保持可变可以避免
    调用方为了"改一个字段"而重建整个对象 —— 本特性没有需要哈希或作字典键的场景。
    """

    role: str
    content: str
    timestamp: int
    message_id: str

    def __post_init__(self) -> None:
        validate_role(self.role)

    @property
    def is_user(self) -> bool:
        return self.role == ROLE_USER

    @property
    def is_assistant(self) -> bool:
        return self.role == ROLE_ASSISTANT

    @property
    def is_system(self) -> bool:
        return self.role == ROLE_SYSTEM

    def as_chat_message(self) -> dict[str, str]:
        """转成 OpenAI messages 数组里的一项。

        ⚠️ **只保留 `role` / `content` 两个键。**

        多传 `timestamp` / `message_id` 会让部分中转站直接拒绝请求
        （OpenAI 兼容接口对 messages 项的额外键处理不一致：有的忽略、
        有的报 400）。而这两个字段对模型没有任何意义 ——
        它们是给服务端日志与排查用的，不该出现在出网请求里。

        这条约束同时是一道**隐私边界**：`message_id` 是服务端内部标识，
        没有理由让它离开本进程。
        """

        return {"role": self.role, "content": self.content}


@dataclass
class SessionMeta:
    """一个会话的元数据。三个字段与 `data-model.md` §2.2 逐项对应。

    ⚠️ `last_active_at` 的语义是"**对话内容**的活跃时间"，
    不是"配置被修改的时间" —— 改 `user_id` MUST NOT 更新它（Q5 裁决 A）。
    若两者混用，这个字段就同时承载两种含义，
    任何基于它的判断（如后续做会话排序）都会在改名后给出误导性的结果。
    """

    user_id: str
    created_at: int
    last_active_at: int
