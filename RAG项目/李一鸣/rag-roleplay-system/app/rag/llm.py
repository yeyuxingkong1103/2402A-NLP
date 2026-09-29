import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class LLMService:
    """OpenAI-compatible chat completion client with a useful local mock mode."""

    def __init__(self, settings: Settings):
        self.provider = settings.llm_provider.lower()
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds

    async def complete(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "mock" or not self.api_key:
            return self._mock_answer(messages)
        payload = {"model": self.model, "messages": messages, "temperature": 0.3}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        logger.info("llm completion received: provider=%s model=%s", self.provider, self.model)
        return str(answer)

    async def stream_complete(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if self.provider == "mock" or not self.api_key:
            answer = self._mock_answer(messages)
            for index in range(0, len(answer), 24):
                yield answer[index : index + 24]
            return

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                        delta = data["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield str(delta)
                    except json.JSONDecodeError:
                        logger.warning("ignored malformed llm stream event")

    @staticmethod
    def _mock_answer(messages: list[dict[str, str]]) -> str:
        user_content = messages[-1]["content"] if messages else ""
        question = user_content.split("用户问题：", 1)[-1].split("\n\n回答要求：", 1)[0].strip()
        context = user_content.split("知识库上下文：", 1)[-1].split("\n\n用户问题：", 1)[0].strip()
        if context.startswith("（本轮没有"):
            return (
                f"我先基于当前角色和已有信息回答：{question}\n\n"
                "知识库暂未提供足够依据，建议补充具体背景、时间和目标，我再帮你进一步分析。"
            )
        first_source = ""
        for line in context.splitlines():
            if line.startswith("【资料"):
                first_source = line
                break
        source_hint = f"\n\n参考：{first_source}" if first_source else ""
        return (
            f"基于当前检索到的资料，关于“{question}”，可以先这样理解：\n\n"
            f"{context.split('】', 1)[-1].strip()[:500]}"
            f"{source_hint}\n\n"
            "这是演示模式生成的回答；接入真实模型后，会结合角色设定和完整上下文生成更自然的回复。"
        )
