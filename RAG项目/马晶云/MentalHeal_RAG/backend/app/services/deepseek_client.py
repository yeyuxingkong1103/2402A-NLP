import json
from collections.abc import Iterator
from typing import Any

import httpx

from app.core.config import Settings
from app.models.chat import AIRole


class DeepSeekClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("your-"):
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    def answer(
        self,
        question: str,
        history: list[dict[str, str]],
        passages: list[dict[str, Any]],
        role: AIRole | None = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(passages, role),
            }
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        response = httpx.post(
            f"{self.settings.deepseek_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.deepseek_model,
                "messages": messages,
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
                "stream": False,
            },
            timeout=self.settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"]).strip()

    def stream_answer(
        self,
        question: str,
        history: list[dict[str, str]],
        passages: list[dict[str, Any]],
        role: AIRole | None = None,
    ) -> Iterator[str]:
        messages = [
            {"role": "system", "content": self._system_prompt(passages, role)},
            *history,
            {"role": "user", "content": question},
        ]
        with httpx.stream(
            "POST",
            f"{self.settings.deepseek_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.deepseek_model,
                "messages": messages,
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
                "stream": True,
            },
            timeout=self.settings.llm_timeout_seconds,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                content = payload.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    yield str(content)

    def _system_prompt(self, passages: list[dict[str, Any]], role: AIRole | None = None) -> str:
        context = []
        remaining = self.settings.rag_context_max_chars
        for item in passages:
            section = (
                f"[{item['title']}，第{item['page_start']}-{item['page_end']}页]\n"
                f"{item['text']}"
            )
            section = section[:remaining]
            if section:
                context.append(section)
                remaining -= len(section)
            if remaining <= 0:
                break
        joined_context = "\n\n".join(context) or "没有检索到足够相关的资料。"
        role_instruction = role.system_instruction if role else "你是温和、稳妥的通用心理健康科普与陪伴助手。"
        return (
            "你是一个只服务于心理健康科普资料的问答助手。\n"
            f"当前角色：{role.name if role else '心理健康助手'}。\n"
            f"角色要求：{role_instruction}\n"
            "请严格根据参考资料回答，不要编造资料中没有的事实。\n"
            "你不能进行正式医学诊断，也不能替代医生、心理咨询师或紧急救助。\n"
            "如果问题超出资料范围，请明确说明不确定性，并建议咨询合格专业人员。\n"
            "回答使用简体中文，语气温和、清晰、非评判。\n\n"
            f"参考资料：\n{joined_context}"
        )
