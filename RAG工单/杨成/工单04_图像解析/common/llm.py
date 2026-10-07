"""LLM 后端与本地证据式回答。"""

from __future__ import annotations

import json
from urllib import request
from urllib.error import HTTPError, URLError

from common.config import Settings
from common.models import Answer, DocumentChunk, SearchResult


class LocalAnswerer:
    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return Answer(answer="上下文中没有找到足够证据。", sources=[])
        evidence = "；".join(result.content for result in results)
        return Answer(answer=f"根据上下文：{evidence}", sources=results)


class OpenAICompatibleAnswerer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return LocalAnswerer().answer(question, results)
        context = "\n".join(
            f"[{result.source} 第{result.page}页] {result.content}" for result in results
        )
        base_url = (self.settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": "只依据给定上下文回答；若上下文没有答案，请明确说明。"},
                {"role": "user", "content": f"问题：{question}\n上下文：\n{context}"},
            ],
        }
        req = request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("API 响应 content 必须是字符串")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            return LocalAnswerer().answer(question, results)
        return Answer(answer=text, sources=results, metadata={"backend": "openai"})


def answer_with_backend(question: str, results: list[SearchResult], settings: Settings | None = None) -> Answer:
    config = settings or Settings.from_env()
    if config.llm_backend == "openai" and config.openai_api_key:
        return OpenAICompatibleAnswerer(config).answer(question, results)
    return LocalAnswerer().answer(question, results)


def chunks_to_results(chunks: list[DocumentChunk]) -> list[SearchResult]:
    return [SearchResult(content=c.content, source=c.source, score=1.0, page=c.page, metadata=c.metadata) for c in chunks]
