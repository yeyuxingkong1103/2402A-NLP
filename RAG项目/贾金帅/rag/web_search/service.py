"""通过 Tavily API 执行联网搜索。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from src import config
from src.core.retrieval.base.retriever_base import RetrievalResult

logger = logging.getLogger(__name__)
_SEARCH_CACHE: dict[str, tuple[float, list[RetrievalResult]]] = {}
_SEARCH_CACHE_LOCK = threading.Lock()


class WebSearchService:
    """Tavily 联网搜索适配器，输出项目统一的检索结果。"""

    name = "web"

    def __init__(self, timeout: Optional[float] = None):
        self.timeout = float(timeout if timeout is not None else config.WEB_SEARCH_TIMEOUT)
        self.cache_ttl = max(0, int(config.WEB_SEARCH_CACHE_TTL))
        self.last_error = ""

    def is_available(self) -> bool:
        return bool(config.TAVILY_API_KEY and config.TAVILY_API_URL)

    @staticmethod
    def _to_result(source: dict[str, Any], rank: int) -> RetrievalResult | None:
        url = str(source.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return None
        title = str(source.get("title") or url).strip()
        text = str(source.get("content") or source.get("snippet") or "").strip()
        if not text:
            return None
        return RetrievalResult(
            content={
                "fusion_key": f"web:{url}",
                "name": title,
                "title": title,
                "document": text,
                "text": text,
                "source_path": url,
                "url": url,
                "domain": urlparse(url).netloc,
                "published_at": str(
                    source.get("published_at")
                    or source.get("published_date")
                    or source.get("date")
                    or ""
                ).strip(),
                "source_type": "web",
                "metadata": {"provider": "tavily", "rank": rank},
            },
            score=1.0 / rank,
            source="web",
            rank=rank,
            reason="Tavily 联网搜索",
            evidence_key=f"web:{url}",
        )

    def search(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        q = (query or "").strip()
        limit = min(max(1, int(top_k or config.WEB_SEARCH_TOP_K)), 5)
        self.last_error = ""
        if not q or not self.is_available():
            self.last_error = "Tavily 联网搜索不可用：请检查 TAVILY_API_KEY 和配置"
            return []

        cache_key = f"{q}\x00{limit}"
        now = time.monotonic()
        with _SEARCH_CACHE_LOCK:
            cached = _SEARCH_CACHE.get(cache_key)
            if cached and self.cache_ttl > 0 and now - cached[0] < self.cache_ttl:
                return list(cached[1])
            if cached:
                _SEARCH_CACHE.pop(cache_key, None)

        try:
            response = requests.post(
                config.TAVILY_API_URL,
                json={
                    "api_key": config.TAVILY_API_KEY,
                    "query": q,
                    "max_results": limit,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            results: list[RetrievalResult] = []
            seen_urls: set[str] = set()
            for source in response.json().get("results", []) or []:
                result = self._to_result(source, len(results) + 1)
                if result is None or result.content["url"] in seen_urls:
                    continue
                seen_urls.add(result.content["url"])
                results.append(result)
                if len(results) >= limit:
                    break
            if not results:
                self.last_error = "Tavily 联网搜索未返回可用来源"
            with _SEARCH_CACHE_LOCK:
                _SEARCH_CACHE[cache_key] = (time.monotonic(), list(results))
            return results
        except requests.Timeout:
            self.last_error = f"Tavily 联网搜索超时（>{self.timeout:.0f}s）"
            logger.warning(self.last_error)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Tavily 联网搜索失败：%s", exc)
        return []
