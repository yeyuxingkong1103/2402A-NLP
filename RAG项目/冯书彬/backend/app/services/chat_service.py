import asyncio
import hashlib
import inspect
import json
import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any

from backend.app.core.config import settings
from backend.app.database.milvus import MilvusUnavailableError, create_milvus_store
from backend.app.embeddings.embedding_factory import get_embedding_client
from backend.app.llm.base import LlmRequest
from backend.app.llm.deepseek_client import DeepSeekClient, DeepSeekUnavailableError
from backend.app.models.conversation import Conversation
from backend.app.models.message import Message
from backend.app.rag.chat_adapter import Task8RagDecider
from backend.app.rag.hybrid_retriever import retrieve_candidates
from backend.app.rag.pipeline import RetrievalDecision
from backend.app.safety.answer_validator import validate_answer
from backend.app.safety.legal_advice_guard import classify_legal_scope, out_of_scope_answer
from backend.app.safety.risk_detector import detect_risk, emergency_guidance
from backend.app.schemas.chat import ChatResult, FactCorrection, ScopeResult
from backend.app.services.privacy_service import redact_pii
from backend.app.services.request_control import RequestLease, create_request_controller

logger = logging.getLogger(__name__)
_MAX_CONTEXT_CHARS_PER_FRAGMENT = 800
_MAX_CONTEXT_CHARS_TOTAL = 2400


class MessageTooLongError(ValueError):
    pass


class RegenerationLimitError(ValueError):
    pass


class ConversationNotFoundError(ValueError):
    pass


def validate_message_length(text: str) -> None:
    # 中文字符按 Python 字符数计算，超过配置上限直接拒绝。
    if len(text) > settings.MESSAGE_MAX_CHARS:
        raise MessageTooLongError("单条消息超过5000中文字符上限")


def classify_scope(text: str) -> ScopeResult:
    # 对外暴露纯函数，便于测试和 API 复用。
    return classify_legal_scope(text)


def _citation_to_dict(citation: Any) -> dict[str, Any]:
    # Citation 是 dataclass；兼容早期 dict 引用，避免默认参数提前求值触发 TypeError。
    if hasattr(citation, "__dict__"):
        return dict(citation.__dict__)
    return dict(citation)


def create_default_rag_decider() -> Task8RagDecider:
    # 只有显式配置 Milvus 时才启用真实向量检索，开发和测试默认保持安全 fallback。
    if not settings.MILVUS_URI:
        return Task8RagDecider()
    embedding_client = get_embedding_client()
    vector_store = create_milvus_store(settings)

    def candidate_provider(text: str):
        from backend.app.rag.result_merger import RetrievalFilters

        return retrieve_candidates(text, RetrievalFilters(), embedding_client=embedding_client, vector_store=vector_store)

    async def legacy_rag_decider(text: str) -> RetrievalDecision:
        from backend.app.rerank.rerank_factory import get_rerank_client

        try:
            return await Task8RagDecider(candidate_provider=candidate_provider, rerank_client=get_rerank_client())(text)
        except MilvusUnavailableError as exc:
            logger.warning("RAG 向量库不可用，使用安全依据不足 fallback", extra={"error_type": type(exc).__name__, "query_chars": len(text)})
            from backend.app.rag.chat_adapter import safe_insufficient_decision

            return safe_insufficient_decision("rag_not_configured")

    async def langchain_rag_decider(text: str) -> RetrievalDecision:
        from backend.app.rerank.rerank_factory import get_rerank_client

        try:
            from backend.app.langchain.rag_adapter import create_langchain_rag_decider

            decider = create_langchain_rag_decider(candidate_provider, get_rerank_client())
            return await decider(text)
        except (MilvusUnavailableError, ImportError, RuntimeError) as exc:
            logger.warning("LangChain RAG 适配层不可用，使用安全依据不足 fallback", extra={"error_type": type(exc).__name__, "query_chars": len(text)})
            from backend.app.rag.chat_adapter import safe_insufficient_decision

            return safe_insufficient_decision("langchain_not_configured")

    framework = settings.RAG_FRAMEWORK.lower()
    if framework == "legacy":
        return legacy_rag_decider
    if framework == "shadow":
        async def shadow_rag_decider(text: str) -> RetrievalDecision:
            legacy_result = await legacy_rag_decider(text)
            _schedule_langchain_shadow(text, legacy_result, candidate_provider)
            return legacy_result

        return shadow_rag_decider
    return langchain_rag_decider


def _schedule_langchain_shadow(text: str, legacy_result: RetrievalDecision, candidate_provider: Callable[[str], Any]) -> None:
    # 同步 API 使用 asyncio.run 后会关闭请求事件循环，shadow 必须脱离该循环继续执行。
    def run_shadow() -> None:
        asyncio.run(_run_langchain_shadow(text, legacy_result, candidate_provider))

    threading.Thread(target=run_shadow, name="rag-langchain-shadow", daemon=True).start()


async def _run_langchain_shadow(text: str, legacy_result: RetrievalDecision, candidate_provider: Callable[[str], Any]) -> None:
    from backend.app.rerank.rerank_factory import get_rerank_client

    try:
        from backend.app.langchain.rag_adapter import create_langchain_rag_decider

        langchain_result = await create_langchain_rag_decider(candidate_provider, get_rerank_client())(text)
        summary = _rag_shadow_summary(legacy_result, langchain_result)
        logger.info("RAG shadow 对比完成 %s", json.dumps(summary, ensure_ascii=True, sort_keys=True))
    except Exception as exc:
        logger.warning("RAG shadow 对比失败", extra={"error_type": type(exc).__name__, "query_chars": len(text)})


def _rag_shadow_summary(legacy_result: RetrievalDecision, langchain_result: RetrievalDecision) -> dict[str, Any]:
    def document_digest(result: RetrievalDecision) -> str:
        identifiers = [f"{item.material_id}:{item.version_id}:{item.id}" for item in result.top_documents]
        return hashlib.sha256("|".join(identifiers).encode("utf-8")).hexdigest()[:16]

    same_decision = (
        legacy_result.can_answer == langchain_result.can_answer
        and legacy_result.reason == langchain_result.reason
        and document_digest(legacy_result) == document_digest(langchain_result)
        and len(legacy_result.citations) == len(langchain_result.citations)
    )
    return {
        "comparison_status": "match" if same_decision else "mismatch",
        "legacy_can_answer": legacy_result.can_answer,
        "langchain_can_answer": langchain_result.can_answer,
        "legacy_reason": legacy_result.reason,
        "langchain_reason": langchain_result.reason,
        "legacy_document_digest": document_digest(legacy_result),
        "langchain_document_digest": document_digest(langchain_result),
        "legacy_citation_count": len(legacy_result.citations),
        "langchain_citation_count": len(langchain_result.citations),
    }


class ChatService:
    def __init__(self, rag_decider: Callable[[str], RetrievalDecision] | None = None, llm_client: Any | None = None, repository: Any | None = None, request_controller: Any | None = None):
        # 依赖可注入，测试不触发真实检索或外部模型。
        self.rag_decider = rag_decider or create_default_rag_decider()
        self.llm_client = llm_client or DeepSeekClient(api_key=settings.DEEPSEEK_API_KEY)
        # repository 为空时继续使用内存态，兼容现有 MVP 测试。
        self.repository = repository
        self.conversations: dict[str, Conversation] = {}
        self.messages: dict[str, Message] = {}
        self._request_controller = request_controller or create_request_controller()
        self._rate_limiter = self._request_controller

    def reset_for_tests(self, rag_decider: Callable[[str], RetrievalDecision] | None = None, llm_client: Any | None = None) -> None:
        # 测试重置依赖和内存状态，避免跨用例污染。
        self.rag_decider = rag_decider or create_default_rag_decider()
        self.llm_client = llm_client or DeepSeekClient(api_key=settings.DEEPSEEK_API_KEY)
        self.conversations.clear()
        self.messages.clear()
        self._request_controller.reset()
        self._rate_limiter = self._request_controller

    async def send_message(self, user_id: str, conversation_id: str, text: str) -> ChatResult:
        # 长度检查必须早于任何外部依赖调用。
        validate_message_length(text)
        limit_result, lease = self._try_enter(user_id, conversation_id)
        if limit_result is not None:
            return limit_result
        try:
            return await self._send_message_inside_limits(user_id, conversation_id, text)
        finally:
            if lease is not None:
                lease.release()

    async def _send_message_inside_limits(self, user_id: str, conversation_id: str, text: str) -> ChatResult:
        conversation = self._get_or_create_conversation(user_id, conversation_id)
        redacted = redact_pii(text)
        message = self._record_user_message(conversation, user_id, redacted.text)
        logger.info("聊天消息进入编排", extra={"conversation_id": conversation_id, "message_id": message.id, "chars": len(text), "pii_types": redacted.redacted_types})
        risk = detect_risk(text)
        if risk.level == "emergency":
            return self._finish_message(message, conversation, "emergency_guidance", emergency_guidance(), [], "emergency_risk")
        scope = classify_scope(text)
        if not scope.in_scope:
            return self._finish_message(message, conversation, "out_of_scope", out_of_scope_answer(), [], scope.reason)
        missing = self._missing_follow_up_fields(text)
        if missing and conversation.follow_up_count < 3 and not self._should_answer_immediately(text):
            conversation.follow_up_count += 1
            return self._finish_message(message, conversation, "follow_up_required", self._build_follow_up_answer(missing, conversation.follow_up_count), [], "missing_facts")
        if missing and conversation.follow_up_count >= 3 and not self._should_answer_immediately(text):
            return self._finish_message(message, conversation, "conditional_analysis", self._build_conditional_answer(missing), [], "follow_up_limit_reached")
        return await self._answer_with_rag_and_llm(message, conversation, redacted.text)

    async def regenerate_answer(self, user_id: str, message_id: str) -> ChatResult:
        # 重新生成基于原用户消息，并重新执行安全、范围、RAG 和引用校验。
        message = self._get_owned_message(user_id, message_id)
        if message.regeneration_count >= 3:
            raise RegenerationLimitError("重新生成次数已达上限")
        message.regeneration_count += 1
        logger.info("重新生成回答", extra={"conversation_id": message.conversation_id, "message_id": message.id, "attempt": message.regeneration_count})
        return await self._regenerate_from_text(message, message.user_text)

    async def correct_facts(self, user_id: str, conversation_id: str, correction: FactCorrection) -> ChatResult:
        # 事实纠正必须由用户确认；旧回答仅标记更正，新回答写入新消息以便审计。
        if not correction.confirmed:
            raise ValueError("事实纠正需要用户确认")
        message = self._get_owned_message(user_id, correction.message_id)
        if message.conversation_id != conversation_id:
            raise ConversationNotFoundError("消息不属于该会话")
        message.corrected = True
        message.status = "corrected"
        corrected_text = f"{message.user_text}\n用户确认的事实更正：{redact_pii(correction.correction).text}"
        conversation = self.conversations[conversation_id]
        corrected_message = self._record_user_message(conversation, user_id, corrected_text)
        corrected_message.source_message_id = message.id
        logger.info("事实纠正后生成新回答", extra={"conversation_id": conversation_id, "old_message_id": message.id, "new_message_id": corrected_message.id})
        return await self._regenerate_from_text(corrected_message, corrected_text)

    def restart_consultation(self, user_id: str, conversation_id: str) -> None:
        # 重新开始仅清理当前会话短期状态，不处理长期记忆导出等后续任务。
        conversation = self.conversations.get(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            return
        for message_id in conversation.message_ids:
            self.messages.pop(message_id, None)
        conversation.message_ids.clear()
        conversation.follow_up_count = 0
        conversation.active = True
        logger.info("会话已重新开始", extra={"conversation_id": conversation_id})

    async def _regenerate_from_text(self, message: Message, text: str) -> ChatResult:
        conversation = self.conversations.get(message.conversation_id)
        if conversation is None and self.repository is not None:
            conversation = self.repository.get_conversation(message.user_id, message.conversation_id)
            if conversation is not None:
                self.conversations[message.conversation_id] = conversation
        if conversation is None:
            raise ConversationNotFoundError("会话不存在或无权访问")
        risk = detect_risk(text)
        if risk.level == "emergency":
            return self._finish_message(message, conversation, "emergency_guidance", emergency_guidance(), [], "emergency_risk")
        scope = classify_scope(text)
        if not scope.in_scope:
            return self._finish_message(message, conversation, "out_of_scope", out_of_scope_answer(), [], scope.reason)
        return await self._answer_with_rag_and_llm(message, conversation, text)

    async def _answer_with_rag_and_llm(self, message: Message, conversation: Conversation, text: str) -> ChatResult:
        decision = await self._call_rag(text)
        if not decision.can_answer:
            answer = "当前知识库依据不足，不能在缺少可靠法律依据时进行个案分析。建议补充婚姻家事相关事实，或咨询12348公共法律服务热线、当地法律援助中心等官方渠道。"
            return self._finish_message(message, conversation, "insufficient_basis", answer, [], decision.reason)
        request = self._build_llm_request(text, decision, message.id)
        if settings.RAG_FRAMEWORK.lower() == "langchain":
            from backend.app.langchain.prompt_adapter import build_langchain_llm_request

            request = build_langchain_llm_request(text, decision, message.id)
        try:
            answer = await self._collect_llm_answer(request)
        except DeepSeekUnavailableError:
            return self._finish_message(message, conversation, "llm_unavailable", "模型服务暂不可用，请稍后重试。", [], "deepseek_unavailable")
        if not validate_answer(answer, decision):
            return self._finish_message(message, conversation, "answer_rejected", "回答未通过引用校验，暂不展示。", [], "citation_validation_failed")
        citations = [_citation_to_dict(citation) for citation in decision.citations]
        return self._finish_message(message, conversation, "answered", self._append_citation_notice(answer, citations), citations, "ok")

    async def _call_rag(self, text: str) -> RetrievalDecision:
        result = self.rag_decider(text)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _collect_llm_answer(self, request: LlmRequest) -> str:
        chunks: list[str] = []
        async for chunk in self.llm_client.stream_chat(request):
            chunks.append(chunk)
        return "".join(chunks)

    def _build_llm_request(self, text: str, decision: RetrievalDecision, trace_id: str) -> LlmRequest:
        system = "你是婚姻家事法律信息助手。只能依据给定资料回答，不能编造依据。"
        context = self._minimize_context(decision.context)
        user = f"用户问题：{text}\n\n可用资料摘要：{context}"
        logger.info("构建最小化模型上下文", extra={"trace_id": trace_id, "context_chars": len(context), "citation_count": len(decision.citations)})
        return LlmRequest(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], citations=decision.citations, trace_id=trace_id, prompt_version="chat-orchestration-v1", citation_validation_passed=True)

    def _minimize_context(self, context: str) -> str:
        # 仅向 DeepSeek 发送截断后的依据片段；日志不得记录完整法律正文。
        fragments = [part.strip()[:_MAX_CONTEXT_CHARS_PER_FRAGMENT] for part in context.split("\n\n") if part.strip()]
        minimized = "\n\n".join(fragments)
        return minimized[:_MAX_CONTEXT_CHARS_TOTAL]

    def _append_citation_notice(self, answer: str, citations: list[dict]) -> str:
        if not citations:
            return answer
        return f"{answer}\n\n参考依据：已通过知识库引用校验。"

    def _should_answer_immediately(self, text: str) -> bool:
        # 已经提出具体法律处理问题时进入 RAG，避免追问阻断依据不足判断。
        concrete_keywords = ["怎么", "如何", "怎么办", "分割", "争取", "起诉", "协议", "诉讼"]
        return any(keyword in text for keyword in concrete_keywords)

    def _missing_follow_up_fields(self, text: str) -> list[str]:
        checks = {
            "所在地省市": ["北京", "上海", "天津", "重庆", "省", "市"],
            "婚姻状态": ["已婚", "结婚", "离婚", "分居"],
            "子女情况": ["孩子", "子女", "抚养"],
            "财产或债务事实": ["财产", "房", "车", "债务", "存款"],
            "当前目标": ["想", "希望", "要求", "争取"],
            "人身危险情况": ["无危险", "安全", "家暴", "威胁", "危险"],
        }
        return [name for name, keywords in checks.items() if not any(keyword in text for keyword in keywords)]

    def _build_follow_up_answer(self, missing: list[str], count: int) -> str:
        fields = "、".join(missing[:4])
        return f"为了避免误导，需要先补充关键信息（第{count}/3轮追问）：请说明{fields}。如存在正在发生的人身危险，请优先报警或求助。"

    def _build_conditional_answer(self, missing: list[str]) -> str:
        fields = "、".join(missing)
        return f"已达到最多3轮追问。已知事实有限，仍缺少：{fields}。这些事实会影响管辖、抚养、财产分割和风险处置判断；只能提供条件化分析，建议补充材料或咨询12348、法律援助中心等官方渠道。"

    def _record_user_message(self, conversation: Conversation, user_id: str, text: str) -> Message:
        message_id = f"msg-{uuid.uuid4().hex}"
        message = Message(id=message_id, conversation_id=conversation.id, user_id=user_id, user_text=text)
        self.messages[message_id] = message
        conversation.message_ids.append(message_id)
        self._save_conversation(conversation)
        self._save_message(message)
        return message

    def _save_conversation(self, conversation: Conversation) -> None:
        if self.repository is not None:
            self.repository.save_conversation(conversation)

    def _save_message(self, message: Message) -> None:
        if self.repository is not None:
            self.repository.save_message(message)

    def _finish_message(self, message: Message, conversation: Conversation, status: str, answer: str, citations: list[dict], reason: str) -> ChatResult:
        message.status = status
        message.answer = answer
        message.citations = citations
        self._save_message(message)
        self._save_conversation(conversation)
        logger.info("聊天编排完成", extra={"conversation_id": conversation.id, "message_id": message.id, "status": status, "reason": reason, "citation_count": len(citations)})
        return ChatResult(status=status, conversation_id=conversation.id, message_id=message.id, answer=answer, citations=citations, follow_up_count=conversation.follow_up_count, reason=reason)

    def _get_or_create_conversation(self, user_id: str, conversation_id: str) -> Conversation:
        conversation = self.conversations.get(conversation_id)
        if conversation is None and self.repository is not None:
            conversation = self.repository.get_conversation(user_id, conversation_id)
            if conversation is not None:
                self.conversations[conversation_id] = conversation
        if conversation is None:
            conversation = Conversation(id=conversation_id, user_id=user_id)
            self.conversations[conversation_id] = conversation
            self._save_conversation(conversation)
        if conversation.user_id != user_id:
            raise ConversationNotFoundError("无权访问该会话")
        return conversation

    def _get_owned_message(self, user_id: str, message_id: str) -> Message:
        message = self.messages.get(message_id)
        if message is None and self.repository is not None:
            message = self.repository.get_message(user_id, message_id)
            if message is not None:
                self.messages[message_id] = message
        if message is None or message.user_id != user_id:
            raise ConversationNotFoundError("消息不存在或无权访问")
        return message

    def _try_enter(self, user_id: str, conversation_id: str) -> tuple[ChatResult | None, RequestLease | None]:
        limit_result, lease = self._request_controller.try_enter(user_id)
        if limit_result is None:
            return None, lease
        return ChatResult("rate_limited", conversation_id, "", limit_result.message, reason=limit_result.reason), None
