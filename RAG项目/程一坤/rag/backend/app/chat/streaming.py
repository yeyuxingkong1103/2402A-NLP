"""流式问答主流程（chat_stream，自 chat/service.py 拆出）。

以 Mixin 形式挂到 ChatService：与同步 chat() 共享检索/记忆/护栏配置，
但流式的护栏衔接方式不同（真流式下已发出的 token 撤不回，方案 A）：
- 引用通过 → 照常结束
- 清单外法规 → 末尾额外 yield 警示行（回答保留）
- 零引用/绝对化表述等不可信回答 → stream_context["replace"] 置为替换全文，
  由 API 层发 replace 事件让前端整段替换

为什么用 Mixin 而不是独立类：流式与同步两条路径必须**共用同一套**检索入参、
拒答阈值、记忆读写这些配置属性。做成独立类就得把 ChatService 的十几个属性
再转发一遍，两边容易各自漂移；Mixin 让它们天然共享 self。
"""
from typing import Any, Iterator

from app.chat.citation_check import (
    check_citations,
    CitationError,
    NoCitationError,
    UncitedConclusionError,
    UnknownLawCitationError,
)
from app.chat.guard import (
    check_absolute_expressions,
    ensure_disclaimer,
    resolve_citation_outcome,
    rewrite_absolute_expressions,
    REFUSAL_ANSWER,
    GREETING_ANSWER,
)
from app.chat.prompt_builder import build_chat_messages
from app.chat.result import ChatResult, _build_context_excerpts
from app.chat.low_score import resolve_low_score
from app.chat.source_assembly import build_citation_sources


class ChatStreamingMixin:
    """ChatService 的流式问答实现（逐 token yield，结束时写 stream_context）。"""

    def chat_stream(
        self,
        question: str,
        *,
        stream_context: dict[str, Any],
        rerank_top_n: int = 5,
        as_of_date: str | None = None,
        jurisdiction: str = "中国大陆",
        document_types: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> Iterator[str]:
        """流式问答：逐块 yield token 文本；结束时把 ChatResult 写入 stream_context["result"]。

        参数：question 用户问题；stream_context **可变字典**，用于把最终 ChatResult
              与 replace 指令带出生成器（生成器无法 return 值给调用方）；
              其余参数与 chat() 同义（检索范围与归属信息）。
        返回：字符串迭代器（增量 token 文本，已发生的护栏追加/替换也在其中）。

        与 chat() 的护栏差异（真流式下已发出的 token 撤不回，方案 A）：
        - 引用通过 → 照常结束
        - 清单外法规 → 末尾额外 yield 警示行（回答保留）
        - 零引用/绝对化表述等不可信回答 → stream_context["replace"] 置为
          替换全文，由 API 层发 replace 事件让前端整段替换
        """
        guardrails_applied = []
        if not self.retrieval_service:
            raise ValueError("retrieval_service 未初始化")

        # 第零步（批次 14）：检索法律知识之前取该用户长期记忆
        # memory_block 与 session_summary 都是 build_chat_messages 的入参，且不依赖
        # retrieval_result，所以统一放在检索调用之前取（顺序固定，便于排查耗时）
        memory_block = self._memory_block(user_id, question)

        # 第零步之二（批次 21）：会话前情（早前轮次压缩摘要）；开关关闭时不读
        session_summary = self._session_summary(user_id, session_id)

        retrieval_result = self.retrieval_service.retrieve(
            question,
            rerank_top_n=rerank_top_n,
            as_of_date=as_of_date,
            jurisdiction=jurisdiction,
            document_types=document_types,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )

        def finish(answer: str, refused: bool, *, include_sources: bool = True) -> None:
            """把收尾结果写进 stream_context["result"]（三条出口共用同一份结构）。

            参数：answer 最终回答全文；refused 是否拒答；include_sources 是否发送本次
            检索法源。寒暄和业务拒答都不携带法源，避免提示语与 citation 相互矛盾。
            返回：None。

            为什么用闭包：拒答、引用失败、正常结束三条出口都要写"同一形状"的结果，
            闭包直接捕获 retrieval_result / guardrails_applied，避免三处重复构造
            导致字段写漏（sources 与 context_excerpts 必须始终同源）。
            """
            # 业务拒答不是异常：拒答文案不应携带低分检索候选，否则前端会同时显示
            # “未找到相关法条”和引用法源，必须与同步 chat() 的 sources 为空保持一致。
            sources = (
                build_citation_sources(retrieval_result.articles)
                if include_sources and not refused
                else []
            )
            context_excerpts = (
                _build_context_excerpts(retrieval_result.articles)
                if include_sources and not refused
                else []
            )
            stream_context["result"] = ChatResult(
                answer=answer,
                # 批次 37：组装收敛到 source_assembly（与 service.py 同源），带条文摘要
                sources=sources,
                refused=refused,
                guardrail_applied=guardrails_applied,
                retrieval_stats=retrieval_result.stats,
                context_excerpts=context_excerpts,
            )

        # 低分结果统一走三档处置，和同步路径共享同一个决策函数。
        retrieval_kwargs = {
            "rerank_top_n": rerank_top_n,
            "as_of_date": as_of_date,
            "jurisdiction": jurisdiction,
            "document_types": document_types,
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
        }
        low_score_decision = resolve_low_score(
            question,
            retrieval_result,
            retrieval_service=self.retrieval_service,
            min_vector_score=self.refusal_min_vector_score,
            retrieve_kwargs=retrieval_kwargs,
        )
        retrieval_result = low_score_decision.retrieval_result
        if low_score_decision.event:
            guardrails_applied.append(low_score_decision.event)
        if low_score_decision.action == "greeting":
            finish(GREETING_ANSWER, False, include_sources=False)
            yield GREETING_ANSWER
            return
        if low_score_decision.action == "refuse":
            finish(REFUSAL_ANSWER, True)
            yield REFUSAL_ANSWER
            return

        # 提示词两条消息：messages[0] 系统提示（角色/口径约束）、messages[1] 用户提示
        # （法条块 + 记忆块 + 会话前情 + 问题）。stream_chat 按这两个下标取值，
        # 改这一处的顺序会直接影响模型看到的约束位置。
        messages = build_chat_messages(
            question,
            retrieval_result.context_block,
            memory_block,
            session_summary,
        )
        # llm_client 的校验放在拒答分支之后：拒答不需要模型，若提到最前面，
        # "未配模型"的环境会连拒答都返回不了（本该是最保底的能力）
        if not self.llm_client:
            raise ValueError("llm_client 未初始化")

        # 边生成边产出（真流式）；同时攒全文供校验与落库
        # parts 与 yield 并行：前端要的是"立刻看到字"，护栏与落库要的是完整文本
        parts: list[str] = []
        for delta in self.llm_client.stream_chat(messages[0]["content"], messages[1]["content"]):
            parts.append(delta)
            yield delta
        answer = "".join(parts)
        # emitted_len 记录"已经发出多少字符"：后续护栏追加的文本只发增量，
        # 否则前端会把已显示的内容再追加一遍（rstrip 是为了对齐"末尾空白不显示"）
        emitted_len = len(answer.rstrip())  # 已 yield 出去的等效长度（不含末尾空白）

        # 引用校验仅适用于有法源的回答；空法源回答允许一般性引导，
        # 但模型若伪造引用仍会在有法源路径中被拦截。
        if retrieval_result.articles:
            try:
                # 允许清单 = 本次真实召回的文档标题；校验器拿它判断回答里的 [n]
                # 是否引到了"没检索到的东西"（越界引用）
                available_sources = [art.document_title for art in retrieval_result.articles]
                check_citations(answer, len(retrieval_result.articles), available_sources)
                guardrails_applied.append("citation_check_passed")
            except CitationError as error:
                # 统一处置阶梯（批次 32 抽取）：分级由异常子类决定，同步/流式共用一份实现。
                # 这里只决定"怎么送达"：警示类回答已发出 token 撤不回，只补发追加的
                # 警示行；硬违规（空回答/越界）整段不可信，让前端 replace 替换全文。
                answer, event = resolve_citation_outcome(answer, error)
                guardrails_applied.append(event)
                if isinstance(error, (NoCitationError, UncitedConclusionError, UnknownLawCitationError)):
                    yield answer[emitted_len:]  # 追加警示行（含空行分隔）
                    emitted_len = len(answer)
                else:
                    stream_context["replace"] = answer
                    finish(answer, False)
                    return

        # 免责声明兜底：提示词已要求模型自带，缺了才补，补的部分作为最后一段发出
        answer = ensure_disclaimer(answer)
        if len(answer) > emitted_len:
            yield answer[emitted_len:]

        # 绝对化表述：流式下已发出的字无法就地改写，改写后全文走 replace
        # （放在免责声明之后，保证 replace 与最终 result.answer 逐字一致）
        if check_absolute_expressions(answer):
            guardrails_applied.append("absolute_expression_rewritten")
            answer = rewrite_absolute_expressions(answer)
            stream_context["replace"] = answer

        # 收尾标记：让日志/前端能确认整条护栏链路已走完（用于区分"护栏全过"
        # 与"流程中途异常退出"）
        guardrails_applied.append("guardrails_applied")
        finish(answer, False)

        # 批次 14：一次问答结束后写入长期记忆（后台线程，不阻塞剩余事件收尾）
        # 放在最后：写记忆依赖最终 answer（可能已被护栏改写），提前写会存错版本
        self._maybe_write_memory(user_id, session_id, question, answer, refused=False)
