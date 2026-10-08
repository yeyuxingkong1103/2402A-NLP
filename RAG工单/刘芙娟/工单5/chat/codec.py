"""消息的序列化。**合规扩展点：将来接加密存储只换这一个对象。**

    MessageCodec  协议（dumps / loads）
    JsonCodec     默认实现

---

## 为什么要有这层抽象

需求十要求「Repository 层的序列化方法做成可替换接口，方便后续接入加密存储」。

具体要防的失效是：如果 `json.dumps` 直接写在 `store.py` 的每个方法里，
那么加密改造**必须改动每一个读写方法** —— 而改动点越多，漏改一处
（导致密文与明文混存在同一个 List 里）的概率越高。而那种混合状态是
**读得出来但看不懂**：一批历史中的几条解不开，没有任何报错指向原因。

抽成 codec 之后，加密实现只需替换一个对象，`store.py` 的方法体一行不改。

## 为什么默认是 JSON 而不是别的

- **`pickle`**：反序列化可执行代码。医疗数据不该引入这种风险，且不可跨语言读。
- **RedisJSON 模块**：需要服务端加载额外模块，而部署前提是"本地 Docker 里的
  标准 Redis"。为序列化格式增加一项部署要求，代价大于收益。
- **JSON**：零部署要求、可读（运维能直接 `redis-cli LINDEX` 看内容）、
  跨语言。缺点是体积略大 —— 在本特性的量级（单会话 50 条消息）下不重要。
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from . import ChatError
from .models import Message, validate_role

__all__ = ["MessageCodec", "JsonCodec"]

# JSON 记录的四个必需字段。与 `Message` 逐项对应。
_FIELDS: tuple[str, ...] = ("role", "content", "timestamp", "message_id")


@runtime_checkable
class MessageCodec(Protocol):
    """消息 ↔ 字符串。**实现 MUST 互逆**：`loads(dumps(m))` 与 `m` 等价。

    用 `Protocol` 而不是抽象基类：扩展实现（加密版）不需要 import 本模块
    就能满足契约，也不必承担继承带来的耦合 —— 这正是"可替换"的意思。
    """

    def dumps(self, message: Message) -> str:
        """序列化。**产出 MUST 是文本**（Redis String 存的就是字节）。"""
        ...

    def loads(self, raw: str) -> Message:
        """反序列化。**无法解析 MUST 抛 `ChatError`，MUST NOT 用默认值兜底。**"""
        ...


class JsonCodec:
    """默认实现：`Message` ↔ JSON 字符串。

    `ensure_ascii=False` 是刻意的：中文内容写成 `\\uXXXX` 会让同一条记录
    膨胀 3–6 倍，而本特性的会话内容几乎全是中文。代价是 `redis-cli` 里
    看到的是 UTF-8 原文 —— 那是优点，不是缺点。
    """

    def dumps(self, message: Message) -> str:
        return json.dumps(
            {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def loads(self, raw: str) -> Message:
        """反序列化。四类失败全部抛错：

            非 JSON / 不是对象 / 缺字段 / 角色非法

        ⚠️ **绝不用默认值兜底。** 这是本类最要紧的一条约束。

        一条解析不了的历史记录若被静默补成"空内容"，模型会收到一条
        「用户说了话但内容是空的」的消息 —— 它不会报错，只会基于残缺的上下文
        给出一个看起来正常的回答。而那一条坏记录**永远不会被任何人发现**。

        抛错的代价是这一次请求失败（可见、可查）；兜底的代价是长期、静默的
        上下文污染。两者不对称，因此选前者。
        """

        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ChatError(
                "历史记录不是合法 JSON（长度 %d）：%s" % (len(raw), type(exc).__name__)
            ) from exc

        if not isinstance(data, dict):
            raise ChatError(
                "历史记录不是 JSON 对象：%s" % type(data).__name__
            )

        missing = [name for name in _FIELDS if name not in data]
        if missing:
            # ⚠️ 只报**字段名**，不报内容 —— 记录里可能含患者隐私（FR-025）。
            raise ChatError("历史记录缺少字段：%s" % "、".join(missing))

        role = data["role"]
        if not isinstance(role, str):
            raise ChatError("历史记录的 role 不是字符串：%s" % type(role).__name__)
        validate_role(role)

        content = data["content"]
        if not isinstance(content, str):
            raise ChatError("历史记录的 content 不是字符串：%s" % type(content).__name__)

        timestamp = data["timestamp"]
        if not isinstance(timestamp, int) or isinstance(timestamp, bool):
            # `bool` 是 `int` 的子类，不排除它会让 `true` 被当成时间戳 1。
            raise ChatError(
                "历史记录的 timestamp 不是整数：%s" % type(timestamp).__name__
            )

        message_id = data["message_id"]
        if not isinstance(message_id, str):
            raise ChatError(
                "历史记录的 message_id 不是字符串：%s" % type(message_id).__name__
            )

        return Message(
            role=role,
            content=content,
            timestamp=timestamp,
            message_id=message_id,
        )
