"""编码与记录组装。**服务端与 CLI 的共用实现。**

⚠️ 本模块 MUST NOT 自己实现编码。

编码口径由 `backend/embed/model.py` 独占（specs/004 的 D3 裁决，该模块的文档
字符串里写得很清楚：批处理脚本与"将来的后端查询"共用本模块）。本模块只做
「拿它来用、组装成一条记录」，**连"薄封装"都不做** —— 薄封装会变成第二个
编码入口，今天它转发，明天有人为了加缓存/日志/重试就在里面写逻辑，
而"口径只有一处"这个保证恰恰是靠没有第二个入口来维持的。
"""

import logging
from datetime import datetime

from backend.embed.model import Encoder

from . import (
    DIM,
    ERR_NOT_EMBEDDED,
    F_ANSWER_ID,
    F_ASKED_AT,
    F_EMBEDDING,
    F_ERROR,
    F_QUESTION,
    F_STATUS,
    F_VECTOR,
    STATUS_FAILED,
    STATUS_OK,
)

logger = logging.getLogger(__name__)

__all__ = ["get_encoder", "reset_encoder", "build_record", "capture_and_embed"]

# 常驻单例。`serve.py` 在启动期调用 `get_encoder()` 把它填上，
# 之后的每个请求直接复用 —— **启动期加载、请求期绝不重复加载**（FR-031）。
#
# 为什么用模块级变量而不是容器/类：本进程只有一个编码器，"全局唯一"这件事
# 用最简单的形式表达即可；引入依赖注入容器去管理一个永远不会变的东西，
# 是为不存在的需求付复杂度。
_encoder: Encoder | None = None


def get_encoder() -> Encoder:
    """返回常驻编码器，首次调用时加载权重。

    ⚠️ **首次调用会阻塞 7–11 秒**（加载 2.27 GB 权重）。服务端 MUST 在启动期
    调用它，MUST NOT 让第一次用户提问来承担这个代价（research R2）。
    """

    global _encoder
    if _encoder is None:
        _encoder = Encoder()
    return _encoder


def reset_encoder() -> None:
    """丢弃常驻编码器。**仅供验收脚本使用**（验证幂等、模拟冷启动）。

    正常路径 MUST NOT 调用 —— 丢掉之后下一次 `get_encoder()` 又要 7–11 秒。
    """

    global _encoder
    _encoder = None


def build_record(
    question: str, answer_id: str, asked_at: datetime, encoder: Encoder
) -> dict:
    """组装一条记录（不含编码）。

    字段顺序由 `store.serialize` 统一重排，这里只需关心内容。

    `embedding` 取自 `encoder.fingerprint` —— **编码这个向量时实际用的那份**，
    不是现算的。二者在正常情况下相同；不同的那一天（比如权重要被替换、
    但 Encoder 已按旧权重构造），我们要记录的是"实际用的"，因为只有它
    能解释这个向量的来历。
    """

    return {
        F_ANSWER_ID: answer_id,
        F_QUESTION: question,
        F_ASKED_AT: asked_at.isoformat(),
        F_EMBEDDING: dict(encoder.fingerprint),
        F_VECTOR: None,
        F_STATUS: STATUS_FAILED,
        F_ERROR: ERR_NOT_EMBEDDED,
    }


def capture_and_embed(
    question: str, answer_id: str, asked_at: datetime | None = None
) -> dict:
    """编码一个问题并组装成完整记录。**服务端与 CLI 共用的唯一入口。**

    **本函数 MUST NOT 抛异常**：编码失败时返回一条 `status=failed` 的完整记录，
    由调用方决定怎么处置（服务端记日志后照常应答；CLI 统计进失败数）。

    为什么不抛：调用方有两条，对失败的反应不同 —— 服务端要"不拖垮请求"，
    CLI 要"报告并继续"。把异常抛出去会逼着两个调用方各写一份 try/except，
    而两份处理迟早会漂。**在这里返回一个统一形状的失败记录，两条路径就都
    只需要判断 `status`。**
    """

    when = asked_at or datetime.now().astimezone().replace(microsecond=0)

    try:
        encoder = get_encoder()
    except Exception as exc:  # noqa: BLE001 —— 模型加载失败，见下方错误字段
        return _failed_record(
            question, answer_id, when, None, "模型不可用：%s" % _brief(exc)
        )

    record = build_record(question, answer_id, when, encoder)

    try:
        vector = encoder.encode_query(question)
    except Exception as exc:  # noqa: BLE001
        return _failed_record(
            question, answer_id, when, encoder, "编码失败：%s" % _brief(exc)
        )

    values = vector.tolist()
    if len(values) != DIM:
        return _failed_record(
            question,
            answer_id,
            when,
            encoder,
            "维度不符：期望 %d，实际 %d" % (DIM, len(values)),
        )

    record[F_VECTOR] = values
    record[F_STATUS] = STATUS_OK
    record[F_ERROR] = None
    return record


def _failed_record(
    question: str,
    answer_id: str,
    asked_at: datetime,
    encoder: Encoder | None,
    reason: str,
) -> dict:
    """构造一条失败记录。

    `encoder` 为 None（模型都没加载起来）时，`embedding` 仍尽力填上
    `fingerprint()` 的结果 —— 它不需要加载模型，能给出"本来打算用哪套口径"。
    拿不到就填 None，并在 reason 里说明。
    """

    if encoder is not None:
        emb: object = dict(encoder.fingerprint)
    else:
        try:
            from backend.embed.model import fingerprint

            emb = fingerprint()
        except Exception:  # noqa: BLE001 —— 指纹都拿不到，只能留空
            emb = None

    return {
        F_ANSWER_ID: answer_id,
        F_QUESTION: question,
        F_ASKED_AT: asked_at.isoformat(),
        F_EMBEDDING: emb,
        F_VECTOR: None,
        F_STATUS: STATUS_FAILED,
        F_ERROR: reason,
    }


def _brief(exc: BaseException) -> str:
    """异常的可读摘要：**类型 + 消息，不含堆栈**。

    `error` 字段会落盘并被读取方展示，堆栈既无用又可能带内部路径
    （constitution 原则 III：日志脱敏）。
    """

    text = str(exc).strip().replace("\n", " ")
    if len(text) > 300:
        text = text[:300] + "…"
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__
