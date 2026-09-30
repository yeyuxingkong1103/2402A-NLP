from __future__ import annotations

"""通义千问兼容 API 客户端，支持普通生成和流式生成。"""

import json
from collections.abc import Iterator

import httpx

from ..config.config import QAConfig


class DashScopeChatClient:
    def __init__(self, config: QAConfig) -> None:
        self.config = config
        if not config.api_key:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    def generate(self, prompt: str) -> str:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = self._payload(prompt)
        payload.pop("stream", None)
        response = httpx.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_seconds,
            trust_env=False,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"通义千问响应缺少 choices：{data}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(f"通义千问响应缺少 message.content：{data}")
        return str(content).strip()

    def stream_generate(self, prompt: str) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=self._payload(prompt),
            timeout=self.config.timeout_seconds,
            trust_env=False,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                try:
                    data = json.loads(value)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield str(content)

    def _payload(self, prompt: str) -> dict:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "你是严谨、可靠的九年级语文教师助手，必须基于检索资料回答。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "stream": True,
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
