"""问答主流程（同步 chat）：串联检索、提示词、LLM、校验、护栏。

完整流程：
1. 检索（调用 retrieval_service）
2. 无候选时直接拒答（不调用 LLM）
3. 组装提示词（调用 prompt_builder）
4. 调用大模型生成回答
5. 引用校验（调用 citation_check）
6. 表述拦截（调用 guard）
7. 确保免责声明
8. 返回结果

职责拆分（本文件只保留同步主流程编排）：
- 结果类型与摘录组装 → app/chat/result.py
- 长期记忆读写钩子 → app/chat/memory_hooks.py
- 流式主流程 chat_stream → app/chat/streaming.py
- 默认依赖装配 build_default_chat_service → app/chat/bootstrap.py

可被 CLI 和 API 层复用。
"""
from typing import Any

from app.chat.citation_check import check_citations, CitationError
from app.chat.guard import (
    apply_guardrails,
    resolve_citation_outcome,
    REFUSAL_ANSWER,
    GREETING_ANSWER,
)
from app.chat.memory_hooks import ChatMemoryMixin
from app.chat.low_score import resolve_low_score
from app.chat.prompt_builder import build_chat_messages
from app.chat.result import ChatResult, _build_context_excerpts
from app.chat.source_assembly import build_citation_sources
from app.chat.streaming import ChatStreamingMixin
from app.retrieval.service import RetrievalService


class ChatService(ChatMemoryMixin, ChatStreamingMixin):
    """问答服务：串联检索、提示词、LLM、校验、护栏。

    Mixin 组成：ChatMemoryMixin（长期记忆读写钩子）、
    ChatStreamingMixin（流式问答 chat_stream）。
    """

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_client: Any | None = None,
        refusal_min_vector_score: float = 0.0,
        long_term_memory: Any | None = None,
        memory_gate: Any | None = None,
        memory_top_k: int = 5,
        memory_write_async: bool = True,
        session_summary_enabled: bool = False,
    ):
        """初始化问答服务。

        Args:
            retrieval_service: 检索服务
            llm_client: 大模型客户端（需要有 chat(system, user) 方法）
            refusal_min_vector_score: 拒答阈值（候选向量相似度下限），
                0 表示不启用；取值由阶段 8.4 评测集校准得出，写在配置里
            long_term_memory: 长期记忆存储（批次 14，可选）；None = 不读不写
            memory_gate: 用户开关校验 callable(user_id) -> bool（接口 8.3）；
                None = 只要有记忆存储就启用读写
            memory_top_k: 注入提示词的记忆条数上限（默认 5）
            memory_write_async: True = 问答结束后用后台线程写记忆，
                不阻塞回答返回；测试置 False 以便同步断言
            session_summary_enabled: 是否把会话前情（早前轮次压缩摘要）注入提示词
                （批次 21，默认 False = 不读不注入，行为与引入前逐字一致）
        """
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client
        self.refusal_min_vector_score = refusal_min_vector_score
        self.long_term_memory = long_term_memory
        self.memory_gate = memory_gate
        self.memory_top_k = memory_top_k
        self.memory_write_async = memory_write_async
        self.session_summary_enabled = session_summary_enabled

    def chat(
        self,
        question: str,
        *,
        vector_recall_limit: int | None = None,
        keyword_recall_limit: int | None = None,
        rerank_candidate_limit: int | None = None,
        rerank_top_n: int = 5,
        as_of_date: str | None = None,
        jurisdiction: str = "中国大陆",
        document_types: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        skip_citation_check: bool = False,
    ) -> ChatResult:
        """执行完整问答流程。

        Args:
            question: 用户问题
            vector_recall_limit: 向量召回条数（None = 读全局配置，默认 40）
            keyword_recall_limit: 关键词召回条数（None = 读全局配置，默认 40）
            rerank_candidate_limit: 融合后送重排的候选上限（None = 读全局配置，默认 40）
            rerank_top_n: 重排后保留条数
            as_of_date: 适用时间点（YYYY-MM-DD）
            jurisdiction: 法域
            skip_citation_check: 是否跳过引用校验（调试用）

        Returns:
            ChatResult: 包含回答、法源、拒答标记、护栏应用情况
        """
        guardrails_applied = []

        # 第零步（批次 14）：检索法律知识**之前**取该用户长期记忆，
        # 与检索并行准备个性化背景（读取失败不影响主流程）
        memory_block = self._memory_block(user_id, question)

        # 第零步之二（批次 21）：会话前情（早前轮次压缩摘要）。
        # 与长期记忆同属"背景"而非"依据"；开关关闭时返回 None，不读 Redis
        session_summary = self._session_summary(user_id, session_id)

        # 第一步：检索
        if not self.retrieval_service:
            raise ValueError("retrieval_service 未初始化")

        retrieval_result = self.retrieval_service.retrieve(
            question,
            vector_recall_limit=vector_recall_limit,
            keyword_recall_limit=keyword_recall_limit,
            rerank_candidate_limit=rerank_candidate_limit,
            rerank_top_n=rerank_top_n,
            as_of_date=as_of_date,
            jurisdiction=jurisdiction,
            document_types=document_types,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )

        # 第二步：低分结果统一走三档处置，避免同步与流式路径行为分叉。
        retrieval_kwargs = {
            "vector_recall_limit": vector_recall_limit,
            "keyword_recall_limit": keyword_recall_limit,
            "rerank_candidate_limit": rerank_candidate_limit,
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
            return ChatResult(
                answer=GREETING_ANSWER,
                sources=[],
                refused=False,
                guardrail_applied=guardrails_applied,
                retrieval_stats=retrieval_result.stats,
            )
        if low_score_decision.action == "refuse":
            return ChatResult(
                answer=REFUSAL_ANSWER,
                sources=[],
                refused=True,
                guardrail_applied=guardrails_applied,
                retrieval_stats=retrieval_result.stats,
            )

        # 第三步：组装提示词（记忆段落与前情段落均可选注入）
        messages = build_chat_messages(
            question,
            retrieval_result.context_block,
            memory_block,
            session_summary,
        )

        # 第四步：调用大模型
        if not self.llm_client:
            raise ValueError("llm_client 未初始化")

        try:
            answer = self.llm_client.chat(messages[0]["content"], messages[1]["content"])
        except Exception as error:
            # LLM 调用失败，返回错误信息
            raise RuntimeError(f"大模型调用失败：{type(error).__name__}: {error}") from error

        # 第五步：有法源时校验引用；无检索结果时没有可引用编号，跳过引用校验。
        # 空法源回答仍会经过免责声明和绝对化表述护栏。
        if not skip_citation_check and retrieval_result.articles:
            try:
                available_sources = [art.document_title for art in retrieval_result.articles]
                check_citations(answer, len(retrieval_result.articles), available_sources)
                guardrails_applied.append("citation_check_passed")
            except CitationError as error:
                # 统一处置阶梯（批次 32 抽取）：分级由异常子类决定，同步/流式共用一份实现
                answer, event = resolve_citation_outcome(answer, error)
                guardrails_applied.append(event)

        # 第六步：应用护栏（表述拦截、免责声明）
        answer = apply_guardrails(answer, retrieval_result.articles)
        guardrails_applied.append("guardrails_applied")

        # 第七步：组装法源列表（批次 37 收敛到 source_assembly，带条文摘要；
        # 此前与 streaming.py 逐字平行，漏改一处就是行为分叉）
        sources = build_citation_sources(retrieval_result.articles)

        # 第八步（批次 14）：一次问答结束后写入长期记忆（后台线程，不阻塞返回）
        self._maybe_write_memory(user_id, session_id, question, answer, refused=False)

        return ChatResult(
            answer=answer,
            sources=sources,
            refused=False,
            guardrail_applied=guardrails_applied,
            retrieval_stats=retrieval_result.stats,
            context_excerpts=_build_context_excerpts(retrieval_result.articles),
        )
