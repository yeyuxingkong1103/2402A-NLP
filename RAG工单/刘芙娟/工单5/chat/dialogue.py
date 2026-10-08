"""单轮对话编排。**本特性的核心，也是宪法原则 II/IV/V 的落点。**

    stream_chat_turn(...) -> AsyncIterator[str]     产出 SSE 文本块

一轮的完整链路（contracts/chat-api.md §3）：

    1. 发 status 首帧
    2. 取上下文窗口（读历史 + 裁剪）        ← 必须在落库用户消息之前
    3. 落库用户消息（脱敏后）
    4. **本轮独立检索**
   4b. **每轮**都用历史改写查询 → 若改写词不同，重新检索并择优
    5. 发 citations 帧
    6. 发前置片段（紧急话术）
    7. 生成 → 发 token 帧                   ← 检索为空则走拒答，跳过
    8. 装配 answer_text（唯一拼接点）
    9. 落库助手消息（**已装配文本**）
    10. 发 done 帧

---

## 步骤 2 为什么必须在步骤 3 之前

窗口是从**已存储的历史**裁出来的。若先落库本轮问题再取窗口，
本轮问题会同时出现在"历史"和"最终提示词"里 —— 模型看到同一句话两遍，
而它会据此认为用户在强调这件事。

## 与 `/ask` 的关系：同一条链路，不是第二套

检索用 `pipeline.retrieve`、生成用 `pipeline.generation_deltas`、
装配用 `pipeline.assemble_answer_text` —— 与 `backend/api/stream.py` 调用的是
**同一批函数**。差别只有两处：输入端多了一份 `history`，输出端多了一次落库。

⚠️ 若这里自己实现一遍检索或生成，`/ask` 的验收就不再能证明本接口的行为 ——
而 CLI 存在的全部意义就是验证服务端跑的那条链路。

## 逐字话术

本模块 **MUST NOT 出现**紧急话术或免责声明的文本 —— 它们由
`pipeline.assemble_answer_text()` 唯一拼接（D3 / FR-017）。
这里只决定"传什么 preamble 列表、正文是什么"，措辞不归它管。
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator

from backend.api import (
    EV_CITATIONS,
    EV_DONE,
    EV_STATUS,
    EV_TOKEN,
    REFUSAL_FALLBACK,
    STATE_ACCEPTED,
)
from backend.api.config import AppConfig
from backend.api.events import sse_event
from backend.api.pipeline import (
    assemble_answer_text,
    citation_payload,
    encode_query,
    generation_deltas,
    preamble_chunks,
    retrieve,
)
from backend.api.schemas import answer_response, refusal_response

from . import ACTION_REPLY
from . import audit
from .rewrite import rewrite_query
from .service import ChatService

logger = logging.getLogger(__name__)

__all__ = ["stream_chat_turn"]


async def stream_chat_turn(
    *,
    session_id: str,
    question: str,
    answer_id: str,
    config: AppConfig,
    service: ChatService,
    query_vector: list[float] | None,
) -> AsyncIterator[str]:
    """产出一轮的 SSE 文本块，直到终帧。

    `query_vector` 由路由层经 `capture_question` 取得（与 `/ask` 同一条路径）。
    本函数 MUST NOT 自己编码 —— 那是第二个编码入口，正是 S8 整篇规格在防的事。

    ⚠️ **本函数不做"会话是否存在"的判定。** 那发生在路由层、流开始**之前**
    （见 `contracts/chat-sse.md` §4 的两条合规路径 —— 本特性选的是"流前判定"，
    理由是客户端能拿到一个标准的 404 错误体，而不是一个已开始的流）。
    """

    # ---- 1. 首帧 ----
    yield sse_event(EV_STATUS, {"answer_id": answer_id, "state": STATE_ACCEPTED})

    # ---- 2. 上下文窗口（**必须在落库本轮问题之前**）----
    #
    # ⚠️ 预算显式取自启动期配置，MUST NOT 用 `get_context_window` 的默认值 ——
    #    那样 `.env` 里的 CHAT_MAX_CONTEXT_TOKENS 就永远不会生效，
    #    而它看起来"配了"，只是没人读。
    history = await service.get_context_window(
        session_id, config.chat_max_context_tokens
    )

    # ---- 3. 落库用户消息（`service.append_message` 内部脱敏）----
    #
    # ⚠️ 先落库再检索：若检索或生成失败，用户的消息**仍然应该在历史里** ——
    # 否则用户会看到自己刚说的话消失了，而"这个问题我问过"这件事
    # 在后续轮次里也会丢失。
    await service.append_message(session_id, "user", question)

    # ---- 4. 本轮独立检索 ----
    #
    # ⚠️ **MUST NOT 复用上一轮的检索结果**（FR-014）。
    # 历史的作用是帮模型理解**问题**，不构成**知识来源**。
    retrieval = retrieve(question, answer_id, config, query_vector)

    # ---- 4b. 改写查询 → 择优（**每轮都做**，理由见 `_maybe_rewrite`）----
    #
    # ⚠️ **这一段 MUST 在 `citations` 帧之前** —— 改写的产物正是要被引用的
    #    那批片段。放在帧之后，用户就会看到"引用是本轮的、答案却是按改写词检索的"。
    improved, rewritten_query = await _maybe_rewrite(
        retrieval, history, question, answer_id, config
    )
    if improved is not None:
        retrieval = improved

    # ---- 5. 引用批次（**只含本轮片段**，FR-019）----
    citations = citation_payload(retrieval)
    yield sse_event(EV_CITATIONS, {"citations": citations})

    # ---- 6. 前置片段（紧急话术；I-04 未实现，当前恒为空）----
    preamble = preamble_chunks(config)
    for chunk in preamble:
        yield sse_event(EV_TOKEN, {"text": chunk})

    # ---- 7. 正文 ----
    body = REFUSAL_FALLBACK
    is_refusal = True
    holder: dict = {}

    if retrieval is not None and not retrieval.is_empty:
        async for delta in generation_deltas(
            question,
            answer_id,
            retrieval.passages,
            config,
            holder,
            history=history,
        ):
            yield sse_event(EV_TOKEN, {"text": delta})

        generated = holder.get("result")
        if generated is not None:
            body = generated.answer_text
            is_refusal = False

    # ---- 8. 装配（**全仓唯一拼接点**）----
    answer_text = assemble_answer_text(preamble, body)

    # ---- 9. 落库助手消息 ----
    #
    # ⚠️ 存的是 `answer_text`（**已装配**的完整文本），不是 `generated.answer_text`
    #    （模型的裸输出）。FR-016：历史里必须是"用户当时实际看到的完整文本"。
    #    两者名字相同、内容不同 —— 这是本特性最容易搞错的一处。
    #
    # ⚠️ **拒答轮次同样落库。** 拒答是多轮场景下的高频路径（R12），
    #    不写会让下一轮模型看到"用户说了话但没有回应"，上下文断裂。
    assistant_message = await service.append_message(
        session_id, "assistant", answer_text
    )

    # ---- 10. 终帧 ----
    envelope = (
        refusal_response(answer_id, answer_text, citations)
        if is_refusal
        else answer_response(answer_id, answer_text, citations)
    )
    payload = envelope.model_dump()
    # 本特性独有的两个字段：前端需要 `session_id` 才能发下一条消息，
    # `message_id` 用于日志关联。前七个字段与 I-01 逐字段一致。
    payload["session_id"] = session_id
    payload["message_id"] = assistant_message.message_id
    # 本轮实际采用了的改写词；没改写/改写没改善时为 None。
    # ⚠️ 回显它是为了**让用户能看出改写对不对** —— 改写可能把"它"理解偏，
    #    不显示的话用户只看到"这次答得不一样了"，无从判断。
    payload["rewritten_query"] = rewritten_query

    audit.record(ACTION_REPLY, session_id, n_messages=None)
    logger.info(
        "chat_turn_done answer_id=%s session_id=%s is_refusal=%s citations=%d "
        "history=%d answer_chars=%d rewritten=%s",
        answer_id,
        session_id,
        is_refusal,
        len(envelope.citations),
        len(history),
        len(answer_text),
        "yes" if rewritten_query else "no",
    )

    yield sse_event(EV_DONE, payload)


def _better(new, old) -> bool:
    """改写后的检索结果是否**严格好于**原来的。

    ⚠️ 判据是"严格更好"，不是"不为空就用"。
    改写可能把问题理解偏，给出一个主题不同但同样有片段的结果 ——
    那种情况下用它，等于用一次改写把一次本来尚可的检索换掉了。

        原结果完全为空   → 新的非空即算改善（有依据好过没依据）
        原结果非空       → 新的 `top_score` 必须**更高**才算改善

    `top_score` 是契约里"排序依据"的对外字段（`docs/05` §4.2），
    用它比较是拿同一把尺子量两个结果 —— 这正是它存在的意义。
    """

    if new is None or new.is_empty:
        return False
    if old is None or old.is_empty:
        return True
    return (new.top_score or 0.0) > (old.top_score or 0.0)


async def _maybe_rewrite(retrieval, history, question, answer_id, config):
    """尝试改写并重新检索。返回 `(更好的结果或 None, 实际采用的改写词或 None)`。

    ---

    ⚠️ **任何一步失败都退回原结果，MUST NOT 抛异常** —— 调用方在流中间，
    异常无法变成状态码，只会让客户端拿到一个中断的流（与"网络断了"不可区分）。

    ⚠️ **改写失败 MUST NOT 降级成"拿历史硬答"。** 那与"检索为空时不拒答"
    是同一件事，只是换了个触发点，且直接违反 constitution 原则 II。
    改写失败 → 原样走拒答。

    ---

    ## ⚠️ 为什么**每轮都试**，而不是只在"检索失败"时试

    初版有一个 `_needs_rewrite()` 门禁：只在 `is_empty` 或 `below_threshold` 时改写。
    **那个门禁被实测推翻了。**

    真实失效长这样（用户实测，2026-10-08）：

        问：血压多少算高血压？   → 正常回答
        问：那怎么预防           → 回答的是**流感**预防

    实测的检索数字：

        查询「那怎么预防」      语义路命中 20 条，top 余弦 **0.6078**
                                （过了阈值 → `is_empty=False`、`below_threshold=False`）
                                关键词路覆盖率 **0.33**（它**正确**地拒绝了这些片段）

    **关键词路拦住了，语义路把住了关。**「那怎么预防」没有主语，BGE-M3 把它编码成一个
    模糊的向量，落点离"流感预防"的文本很近 —— 于是它**长得像一次成功的检索**。

    门禁假设的失效是"检索为空"；而实际的失效是"**检索到了错的东西**"。
    后者在 `RetrievalResult` 上与成功**没有区别**，门禁天然看不见。

    同一句话的两个检索结果（实测）：

        原查询「那怎么预防」     top 余弦 0.6078   覆盖率 0.33
        改写「高血压 怎么预防」   top 余弦 0.7302   覆盖率 0.67

    两个信号一致 —— **所以让改写每轮都跑，`_better` 就会正确采用它。**

    ## 代价（明写）

    **每轮 +1 次模型调用**（约 1~2 秒）。这是为"追问不再跑偏"付的确定代价。

    ⚠️ 缓解：`rewrite_query` 在模型判断"本轮问题已自足"时会**原样返回**，
    那时本函数直接返回 `None, None` —— **不会多一次检索**，只有那一次模型调用。
    提示词的第 3 条规则（"已经完整就原样返回"）因此不再是可选的礼貌，
    它直接决定正常提问要多花多少时间。
    """

    try:
        rewritten = await rewrite_query(history, question, config)
    except Exception as exc:  # noqa: BLE001 —— 见上
        logger.error(
            "chat_rewrite_error answer_id=%s error=%s", answer_id, type(exc).__name__
        )
        return None, None

    if not rewritten:
        return None, None

    # ⚠️ 改写后的词需要**它自己的向量** —— `query_vector` 是本轮原文的，
    #    拿它去查改写词等于没改写。
    vector = encode_query(rewritten)
    if vector is None:
        return None, None

    fresh = retrieve(rewritten, answer_id, config, vector)
    if not _better(fresh, retrieval):
        # 改写没能改善 —— **不告诉用户**。
        # 回显"已按 X 检索"却给出原来的结果，比不回显更误导。
        logger.info(
            "chat_rewrite_no_gain answer_id=%s rewritten_len=%d", answer_id, len(rewritten)
        )
        return None, None

    logger.info("chat_rewrite_applied answer_id=%s rewritten_len=%d", answer_id, len(rewritten))
    return fresh, rewritten


def new_answer_id() -> str:
    """生成一轮回答的标识。路由层在流开始之前调用。

    与 `/ask` 同一取向（`backend/api/routes.py`）：它要同时出现在首帧
    与服务端日志，生成点必须早于流的启动。
    """

    return str(uuid.uuid4())
