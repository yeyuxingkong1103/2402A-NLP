from __future__ import annotations

import time

import httpx

from ..config import Settings
from .llm import require_siliconflow_key


class RerankClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_error = ""

    def rank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        return self.rerank(query, docs)

    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        docs = docs[: max(0, int(getattr(self.settings, "reranker_max_docs", 50) or 50))]
        if len(docs) <= 1:
            return [(0, 1.0)] if docs else []
        try:
            require_siliconflow_key(self.settings)
            retries = int(getattr(self.settings, "rerank_retry_count", 2) or 2)
            delay = float(getattr(self.settings, "rerank_retry_delay", 1.0) or 1.0)
            for attempt in range(retries + 1):
                try:
                    response = httpx.post(f"{self.settings.siliconflow_base_url.rstrip('/')}/rerank", headers={"Authorization": f"Bearer {self.settings.siliconflow_api_key}"}, json={"model": self.settings.reranker_model, "query": query, "documents": docs, "top_n": min(self.settings.reranker_top_n, len(docs))}, timeout=getattr(self.settings, "reranker_timeout", 60))
                    response.raise_for_status()
                    self.last_error = ""
                    return [(row["index"], float(row["relevance_score"])) for row in response.json().get("results", [])]
                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    if attempt >= retries:
                        raise exc
                    time.sleep(delay * 2 ** attempt)
        except Exception as exc:
            self.last_error = f"重排失败，已降级: {str(exc)[:200]}"
        return [(index, 0.5) for index in range(len(docs))]
