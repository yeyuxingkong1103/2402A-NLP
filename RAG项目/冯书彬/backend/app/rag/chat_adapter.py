import inspect
import logging
from collections.abc import Callable
from typing import Any

from backend.app.rag.pipeline import RetrievalDecision, rerank_and_select

logger = logging.getLogger(__name__)


class Task8RagDecider:
    # 适配 Task 8 pipeline；未注入真实候选源和 reranker 时只走安全 fallback。
    def __init__(self, candidate_provider: Callable[[str], Any] | None = None, rerank_client: Any | None = None):
        self.candidate_provider = candidate_provider
        self.rerank_client = rerank_client

    async def __call__(self, text: str) -> RetrievalDecision:
        if self.candidate_provider is None or self.rerank_client is None:
            logger.warning("RAG 未配置真实知识库，使用安全依据不足 fallback", extra={"query_chars": len(text)})
            return safe_insufficient_decision("rag_not_configured")
        candidates = self.candidate_provider(text)
        if inspect.isawaitable(candidates):
            candidates = await candidates
        return rerank_and_select(text, list(candidates), self.rerank_client)


def safe_insufficient_decision(reason: str = "insufficient_legal_basis") -> RetrievalDecision:
    # 安全 fallback 只用于未配置真实检索时，生产应注入 Task8RagDecider 的真实依赖或等价 decider。
    return RetrievalDecision(can_answer=False, reason=reason, top_documents=[], citations=[], context="", scores=[])
