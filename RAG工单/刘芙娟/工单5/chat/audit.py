"""会话操作的审计日志。

    record(action, session_id, user_id, n_messages)

输出一行结构化键值对，供日志系统采集。**不含消息内容。**

---

## 审计日志与运行日志的分工

| | 运行日志（`service` / `dialogue` 里的 `logger.info`） | 审计日志（本模块） |
|---|---|---|
| 回答的问题 | "这次请求怎么了" | "谁对哪个会话做了什么" |
| 字段 | session_id / message_id / 长度 / 耗时 | action / session_id / user_id / 时间 |
| 读者 | 排查故障的人 | 合规审查 |

分开是因为两者的**留存期望不同**：审计日志通常要按合规要求留存更久，
而运行日志可以用调试级别整体调低。混在一起会让"调低日志级别"变成
"把审计记录也关掉"。

## ⚠️ 绝不记录的内容

- **消息内容**（FR-025）—— 本模块的任何调用点 MUST NOT 把 content 传进来。
  签名里没有这个参数，是刻意的：让"误传内容"在类型层面就不可能。
- **患者隐私字段** —— 落库前已由 `redact.py` 处理，但审计日志读的是
  调用方传入的值，不经过脱敏链路。因此 `user_id` 的取值 MUST 由
  `docs/05` 约束为"调用方自定的显示名"，MUST NOT 用于传递实名信息。
"""

from __future__ import annotations

import logging
import time

__all__ = ["record"]

# 独立的 logger 名。
#
# ⚠️ 用 `backend.chat.audit` 而不是复用 `backend.chat`：部署时可能要给
# 审计日志单独配一个 handler（写到单独的文件、单独设留存期），
# 而 logger 名是配置能匹配到的唯一抓手。
_audit_logger = logging.getLogger(__name__ + ".audit")


def record(
    action: str,
    session_id: str,
    *,
    user_id: str | None = None,
    n_messages: int | None = None,
) -> None:
    """记一条审计记录。

    `action` 取 `backend/chat/__init__.py` 的 `ACTION_*` 闭集。
    `user_id` 与 `n_messages` 可选 —— 删除会话时元数据已经没了，
    取不到 `user_id` 属正常，MUST NOT 为了凑齐字段去回读一次 Redis。

    ⚠️ **参数里没有 `content`** —— 见模块文档。
    """

    # 时间戳取整秒：毫秒精度对"谁在什么时候做了什么"没有增益，
    # 却会让同一秒内的多条记录看起来杂乱。
    fields = {
        "action": action,
        "session_id": session_id,
        "ts": int(time.time()),
    }
    if user_id is not None:
        fields["user_id"] = user_id
    if n_messages is not None:
        fields["n_messages"] = n_messages

    # 手工拼键值对而不是 `json.dumps`：一条审计记录应当是**单行可读文本**，
    # 而 JSON 的花括号与引号在 `grep` / `awk` 里都要转义。
    # 值里的空白替换成下划线，避免破坏"一行一条、空格分列"的结构。
    parts = " ".join(
        "%s=%s" % (key, str(value).replace(" ", "_"))
        for key, value in fields.items()
    )
    _audit_logger.info("chat_audit %s", parts)
