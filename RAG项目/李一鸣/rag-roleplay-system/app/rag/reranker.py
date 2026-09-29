import logging
import re

from app.core.config import Settings
from app.rag.retrievers import tokenize
from app.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankerService:
    """BGE-reranker adapter with lexical scoring as a deterministic fallback."""

    def __init__(self, settings: Settings):
        self.enabled = settings.reranker_enabled
        self.model_name = settings.reranker_model
        self._model = None

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if self.enabled:
            model_scores = self._try_model(query, candidates)
            if model_scores is not None:
                for item, score in zip(candidates, model_scores):
                    item.rerank_score = float(score)
                    item.score = float(score)
                candidates.sort(key=lambda item: item.score, reverse=True)
                logger.info("reranking complete with model: %s candidates", len(candidates))
                return candidates[:top_n]

        query_tokens = set(tokenize(query))
        for item in candidates:
            text_tokens = set(tokenize(item.chunk.text))
            overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
            item.rerank_score = 0.65 * overlap + 0.35 * item.score
            item.score = item.rerank_score
        candidates.sort(key=lambda item: item.score, reverse=True)
        logger.info("reranking complete with fallback scorer: %s candidates", len(candidates))
        return candidates[:top_n]

    def _try_model(self, query: str, candidates: list[RetrievedChunk]) -> list[float] | None:
        try:
            if self._model is None:
                from FlagEmbedding import FlagReranker  # type: ignore

                self._model = FlagReranker(self.model_name, use_fp16=False)
            pairs = [[query, candidate.chunk.text] for candidate in candidates]
            return [float(score) for score in self._model.compute_score(pairs)]
        except Exception:
            logger.exception("BGE reranker unavailable; using fallback scorer")
            return None
