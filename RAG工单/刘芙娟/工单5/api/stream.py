"""事件序列生成器。**全系统的接缝。**

    输入 → 向量化(S8) → 检索(S9) → 提示词+生成(S10) → 装配 → 输出

前两步在 `routes.py` 的上游完成，本模块负责后三步，并把结果按 SSE 送出去。
**这是 I-07（装配）当前唯一的落点** —— `docs/05` §4.4 要求拼接逐字话术的位置
全仓只有一处，那个"一处"就是下面 `answer_text` 的拼装行。

产出的事件序列（契约见 specs/006 的 contracts/sse.md）：

    status  →  citations  →  token*  →  done

## 正文的两条来源

    检索命中 → 流式生成（`token` 多条，逐段追加）
    其余情况 → 单条兜底话术

"其余情况"覆盖：检索为空、全部低于阈值、检索未能进行、生成失败、生成结果
**引用校验不通过**。它们在用户看来是同一件事（这次没东西看），因此共用同一条
路径 —— 但**日志里分别可查**，否则一次生成故障会看起来像一次正常的空结果。
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from . import (
    EV_CITATIONS,
    EV_DONE,
    EV_STATUS,
    EV_TOKEN,
    REFUSAL_FALLBACK,
    STATE_ACCEPTED,
)
from .config import AppConfig
from .events import sse_event
from .pipeline import (
    assemble_answer_text,
    citation_payload,
    generation_deltas,
    preamble_chunks,
    retrieve,
)
from .schemas import answer_response, refusal_response

logger = logging.getLogger(__name__)

__all__ = ["stream_answer"]


async def stream_answer(
    *,
    question: str,
    answer_id: str,
    config: AppConfig,
    query_vector: list[float] | None,
) -> AsyncIterator[str]:
    """产出 SSE 文本块，直到终帧。

    参数：
        question:     已校验并去除首尾空白的问题原文（用于检索与日志）。
        answer_id:    本次回答的唯一标识，由路由层生成。
        config:       启动期配置。从 `request.app.state` 取，**不在此处读环境变量**。
        query_vector: 本次提问的查询向量，由 `capture.py` 在留存时顺带算出
                      （S9 / research R8）。**编码失败时为 `None`**。
                      本函数 MUST NOT 自己编码（FR-003：编码口径只有一处）。

    为什么 `answer_id` 由路由层生成而不是本函数：它需要出现在 `status` 首帧
    **和**服务端日志两侧，生成点必须早于流的启动。放进生成器会让"日志里已记下
    answer_id，但流还没开始产出"这个窗口变得不确定。
    """

    # 是否已产出终帧。用于区分"流正常结束"与"流被中途掐断"。
    #
    # ⚠️ 它必须在**产出 `done` 时就置位**，而不是在循环结束后。
    #    生成器的代码只在被 `__anext__()` 拉动时才执行：消费方拿到最后一个事件
    #    后若不再拉取就直接关闭生成器，循环之后的任何代码**根本不会运行**。
    #    若把"已完成"的判定放在循环之后，这种情况会被误判成断连。
    completed = False

    try:
        # 首帧。契约要求它 MUST 是第一个事件，理由见 research.md R10：
        # 流中断时终帧到不了，answer_id 若只在终帧，这次提问就失去了可关联标识。
        yield sse_event(
            EV_STATUS, {"answer_id": answer_id, "state": STATE_ACCEPTED}
        )

        # 检索（I-05）在此发生：`status` 之后、`citations` 之前。
        #
        # ⚠️ 顺序不能变。把检索提到 `status` 之前会让首帧延迟到检索完成之后，
        #    而首帧的作用恰恰是"请求已被接受"这个即时信号；把它放到
        #    `citations` 之后则 citation 无处可发。
        retrieval = retrieve(question, answer_id, config, query_vector)

        # 引用批次。**即使为空也仍然发送** —— 省略会让前端的 citations 空态分支
        # 从"已被执行过"变成"从未执行"，而那正是 S7 刻意避免的（契约 §4.1）。
        #
        # ⚠️ passages 的降序排序 MUST 在服务端完成（docs/05 §4.2），前端 MUST NOT
        #    重排 —— 这里给出的就是最终顺序。
        #
        # ⚠️ 这份 payload **同时**要进终帧（见下方 `citations=citations`）。
        #    两处各构造一次会让它们迟早不一致，而不一致的表现是引用区在终帧到达
        #    那一刻被清空 —— 前端 `renderDone` 用终帧的 citations 重渲染。
        citations = citation_payload(retrieval)
        yield sse_event(EV_CITATIONS, {"citations": citations})

        # 前置片段（FR-030：紧急话术必须是第一个 token）。
        #
        # ⚠️ 目前恒为空 —— I-04（紧急判定）未实现，只有验收注入 `test_preamble`
        #    时才有内容。把它做成一个**独立的、先于正文的**步骤，是为了让
        #    「话术必须在前」成为数据结构上的顺序，而不是靠一个 if 的位置正确。
        preamble = preamble_chunks(config)
        for chunk in preamble:
            yield sse_event(EV_TOKEN, {"text": chunk})

        # 正文：命中则流式生成，否则兜底话术。
        body = REFUSAL_FALLBACK
        is_refusal = True
        holder: dict = {}

        if retrieval is not None and not retrieval.is_empty:
            async for delta in generation_deltas(
                question, answer_id, retrieval.passages, config, holder
            ):
                yield sse_event(EV_TOKEN, {"text": delta})

            generated = holder.get("result")
            if generated is not None:
                body = generated.answer_text
                is_refusal = False

        # 终帧：逐字段复用 docs/05 §3.1.3。
        #
        # ⚠️ **拼装落在 `pipeline.assemble_answer_text()`，全仓唯一一处**
        #    （docs/05 §3.1.5 / §4.4）。本行原先直接内联那次拼接；
        #    specs/010 把它抽成函数，因为 chat 也要拼同一串 ——
        #    在那边再写一遍就等于出现第二处，而「全仓只应命中一处」这条
        #    可测试的约束随即失效，且**不会有任何测试失败**。
        #
        #    放在服务端而不是前端拼，是为了让「话术是第一句 / 声明是最后一句」
        #    这两条断言能针对**单个字符串**完成 —— 由前端拼的话，这条断言会
        #    分裂在两侧，服务端测试全绿而用户看到的第一句可能不是那句话。
        answer_text = assemble_answer_text(preamble, body)

        envelope = (
            refusal_response(answer_id, answer_text, citations)
            if is_refusal
            else answer_response(answer_id, answer_text, citations)
        )

        # 先置位、先记录，**再** yield。
        # 顺序反过来（yield 之后）就会受"消费方是否再拉一次"影响，
        # 而这正是上面 `completed` 注释里说的那个坑。
        completed = True
        logger.info(
            "回答完成 answer_id=%s is_refusal=%s citations=%d answer_chars=%d",
            answer_id,
            envelope.is_refusal,
            len(envelope.citations),
            len(body),
        )
        yield sse_event(EV_DONE, envelope.model_dump())

    except (asyncio.CancelledError, GeneratorExit) as exc:
        # 客户端断开（关页面 / 刷新 / 服务停止）。
        #
        # 记录但**不当作错误** —— 它解释了为什么这次提问在日志里没有 done 帧。
        # 本期处理它近乎免费；生成模块接入后，"用户关掉页面"会表现为一次
        # 6 秒的 LLM 调用白烧，那时这条日志就是唯一的线索。
        #
        # ⚠️ 为什么同时捕 `GeneratorExit`：
        #
        # 这两个异常来自**两条不同的取消路径**：
        #   · `asyncio.CancelledError` —— `StreamingResponse` 所在的任务被取消
        #     （真实断连、服务停止走这条）；
        #   · `GeneratorExit` —— 生成器被 `aclose()` 关闭（显式收尾、async
        #     上下文管理器退出走这条）。
        #
        # 只捕前者会让后一条路径**静默消失**：日志里既没有 done、也没有
        # client_disconnected，看起来就像这次提问从未发生过。
        #
        # MUST 重新抛出：吞掉它们会让 asyncio 无法正确取消任务、让生成器
        # 在已关闭后继续运行。重新抛出不属于"未捕获异常"—— 它是取消协议的
        # 正常部分。
        #
        # ⚠️ 只在**没走完**的情况下记为断连。
        #    已产出终帧后再被关闭，只是生成器被收尾（GC、显式 aclose），
        #    不是客户端跑了。两者混在一起会让日志里出现大量假的断连记录，
        #    把真正需要排查的那些淹没掉。
        if completed:
            logger.debug("流已完成后被收尾关闭 answer_id=%s", answer_id)
        else:
            logger.info(
                "client_disconnected answer_id=%s reason=%s",
                answer_id,
                type(exc).__name__,
            )
        raise
