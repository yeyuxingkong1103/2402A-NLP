"""提问采集的接缝。

**职责边界**：`routes.py` 只做 HTTP ↔ 模型的协议转换（specs/006 FR-015 的约束），
"把提问留存下来"这件事不是协议转换，因此单独放在这里。`routes.py` 对它的调用
是一行，且**不关心它成功与否**。

---

⚠️ 与 constitution「代码 MUST NOT 以 try/except 吞掉失败条件」的关系：

这里有张力，但不是冲突 —— **被禁止的是"吞掉"（静默、不可观测），不是"捕获"。**

本模块捕获落盘异常后，MUST 留下一条 `capture_failed` 日志（含 `answer_id` 与
异常类型），使这次提问"没有被留存"这件事**事后可查**。区分标准就一条：

    事后能不能查出来这次提问没有被留存？
    能 —— 不是吞掉；不能 —— 是吞掉。

**为什么不让提问失败**：用户来是为了问答，不是为了帮系统写日志。为一次落盘
失败把一个能正常回答的请求打成 500，是把系统的内部维护成本转嫁给用户。
（specs/007 FR-011）
"""

import logging
from datetime import datetime

from backend.query import service
from backend.query import store

logger = logging.getLogger(__name__)

__all__ = ["capture_question"]


def capture_question(question: str, answer_id: str) -> list[float] | None:
    """留存一次提问，并返回本次提问的查询向量。**MUST NOT 向调用方抛出异常。**

    参数：
        question:  已校验并去除首尾空白的问题原文（`validate.py` 的返回值）。
        answer_id: 本次回答的唯一标识，与 SSE 帧、服务端日志同值。

    返回：
        编码成功 → 1024 个 float 的查询向量；
        编码失败 → `None`（同时已留下 `capture_degraded` WARNING 日志）。

    编码与组装交给 `query.service` —— **与 CLI 共用同一份实现**（FR-012/FR-032），
    本模块只负责"调用它、把结果写下去、失败时留下痕迹"。

    `service.capture_and_embed` 约定不抛异常（编码失败时返回 `status=failed`
    的记录），因此这里的 try 实际上只兜住**写盘**这一步。

    ---

    ⚠️ **为什么返回值从 `None` 改成向量**（S9 / research R8）：

    检索（I-05）需要查询向量，而**它刚刚在这里算过** —— S8 只是把它写进留存
    文件就丢掉了。让检索自己再编码一次会同时造成两件事：每次提问多
    100–300 ms 的 CPU 推理，以及**第二个编码入口**。后者正是 S8 整篇规格在防
    的东西（`backend/embed/model.py` 是"全项目唯一的编码口径定义"）。

    ⚠️ **唯一的契约没有变**：本函数仍然 MUST NOT 向调用方抛异常。

    ⚠️ **落盘失败时仍然返回向量。** 留存与检索是两个独立的失败域：
    留存的目的是"可回看"，检索的目的是"当场回答"。让一次落盘失败连带把检索
    也废掉，等于用系统的内部维护成本惩罚用户 —— 而用户的提问本身没毛病。
    这条路径下 `capture_failed` 日志照记，失败仍然可查。
    """

    vector: list[float] | None = None

    try:
        asked_at = datetime.now().astimezone().replace(microsecond=0)
        record = service.capture_and_embed(question, answer_id, asked_at)

        # 先把向量取出来：**写盘之前**。写盘若抛异常，向量仍在手上。
        candidate = record.get("vector")
        if isinstance(candidate, list):
            vector = candidate

        path = store.append_record(record)
        if record.get("status") != "ok":
            # 编码失败但记录写下去了：这是**可重跑的工作队列**（`embed` 子命令
            # 会捡起它），但此刻必须让人看得见，否则用户会以为一切正常。
            logger.warning(
                "capture_degraded answer_id=%s reason=%s",
                answer_id,
                record.get("error"),
            )
        logger.debug("capture_ok answer_id=%s file=%s", answer_id, path.name)
    except Exception as exc:  # noqa: BLE001 —— 见模块文档字符串
        # 只记类型与消息，**不记堆栈全文**（constitution 原则 III：日志脱敏）。
        # 消息里可能带文件路径，而路径对本项目不敏感（不是密钥），保留它
        # 才能定位"目录不存在"还是"权限不足"。
        logger.error(
            "capture_failed answer_id=%s error=%s: %s",
            answer_id,
            type(exc).__name__,
            exc,
        )

    return vector
