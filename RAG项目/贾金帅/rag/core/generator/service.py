"""RAG 回答生成服务：只处理提示词、上下文和业务输出解析。 """

from __future__ import annotations

import logging
import re
from typing import Iterator, Optional

from src.core.generator.prompt import (
    NO_RESULT_PROMPT,
    parse_answer_json,
    truncate_context,
)
from src.core.fusion.prompt import render_hybrid_answer_prompt
from src.model.llm import ModelClient, get_default_model

logger = logging.getLogger(__name__)


class _JSONAnswerStreamParser:
    """Incrementally expose only the JSON ``answer`` string value."""

    _ANSWER_START_RE = re.compile(r'"answer"\s*:\s*"')
    _ESCAPES = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }

    def __init__(self) -> None:
        self.buffer = ""
        self.index = 0
        self.started = False
        self.done = False
        self.escaped = False
        self.unicode_digits: str | None = None
        self.text_parts: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def feed(self, chunk: str) -> str:
        if not chunk or self.done:
            return ""
        self.buffer += chunk
        if not self.started:
            match = self._ANSWER_START_RE.search(self.buffer)
            if match is None:
                return ""
            self.started = True
            self.index = match.end()

        visible: list[str] = []
        while self.index < len(self.buffer) and not self.done:
            char = self.buffer[self.index]
            self.index += 1

            if self.unicode_digits is not None:
                self.unicode_digits += char
                if len(self.unicode_digits) == 4:
                    try:
                        decoded = chr(int(self.unicode_digits, 16))
                    except ValueError:
                        decoded = "\\u" + self.unicode_digits
                    visible.append(decoded)
                    self.text_parts.append(decoded)
                    self.unicode_digits = None
                continue

            if self.escaped:
                self.escaped = False
                if char == "u":
                    self.unicode_digits = ""
                    continue
                decoded = self._ESCAPES.get(char, char)
                visible.append(decoded)
                self.text_parts.append(decoded)
                continue

            if char == "\\":
                self.escaped = True
            elif char == '"':
                self.done = True
            else:
                visible.append(char)
                self.text_parts.append(char)

        return "".join(visible)


class RAGGenerator:
    """使用共享模型生成 RAG 结构化答案。"""

    def __init__(self, model_client: ModelClient | None = None) -> None:
        self.model_client = model_client or get_default_model()

    @property
    def protocol(self) -> str:
        return self.model_client.protocol

    @property
    def model(self) -> str:
        return self.model_client.model

    def is_available(self) -> bool:
        return self.model_client.is_available()

    def chat(self, system: str = "", user: str = "", images=None, **kwargs) -> str:
        return self.model_client.chat(system=system, user=user, images=images, **kwargs)

    async def chat_async(self, system: str = "", user: str = "", images=None, **kwargs) -> str:
        return await self.model_client.chat_async(
            system=system, user=user, images=images, **kwargs
        )

    def generate_answer(
        self,
        query: str,
        context: str,
        intent: Optional[str] = None,
        images=None,
        history: str = "",
    ) -> dict:
        prompt_context = truncate_context(context) if context.strip() else context
        system, user = self._render(query, prompt_context, intent, history)
        try:
            raw = self.model_client.chat(system=system, user=user, images=images)
            return self._parse_with_valid_references(raw, prompt_context)
        except Exception as exc:
            return self._fallback(context, exc)

    def generate_answer_stream(
        self,
        query: str,
        context: str,
        intent: Optional[str] = None,
        images=None,
        history: str = "",
    ) -> Iterator[dict]:
        """Stream visible answer text while retaining the final JSON contract."""
        prompt_context = truncate_context(context) if context.strip() else context
        system, user = self._render(query, prompt_context, intent, history)
        parser = _JSONAnswerStreamParser()
        raw_parts: list[str] = []
        streamed_parts: list[str] = []

        try:
            for chunk in self.model_client.stream(
                user,
                system=system,
                images=images,
                use_thinking=False,
            ):
                if chunk.get("type") != "content":
                    continue
                raw_text = str(chunk.get("text", ""))
                if not raw_text:
                    continue
                raw_parts.append(raw_text)
                visible = parser.feed(raw_text)
                if visible:
                    streamed_parts.append(visible)
                    yield {"type": "token", "text": visible}

            raw = "".join(raw_parts).strip()
            if not raw:
                raise RuntimeError("大模型没有返回最终答案")
            answer = self._parse_with_valid_references(raw, prompt_context)
            answer_text = str(answer.get("answer", ""))
            streamed_text = "".join(streamed_parts)

            # Non-JSON model output cannot be incrementally isolated safely;
            # return it once at completion instead of exposing JSON syntax.
            if not streamed_text and answer_text:
                yield {"type": "token", "text": answer_text}
            elif answer_text.startswith(streamed_text) and len(answer_text) > len(streamed_text):
                yield {"type": "token", "text": answer_text[len(streamed_text):]}
            yield {"type": "result", "answer": answer, "complete": True}
        except Exception as exc:
            logger.warning("RAG 流式回答生成失败: %s", exc)
            partial = "".join(streamed_parts) or parser.text
            if partial:
                answer = {
                    "answer": partial,
                    "items": [],
                    "confidence": 0.0,
                    "references": [],
                }
            else:
                answer = self._fallback(context, exc)
                if answer["answer"]:
                    yield {"type": "token", "text": answer["answer"]}
            yield {
                "type": "error",
                "message": "回答生成中断，请稍后再试。",
            }
            yield {"type": "result", "answer": answer, "complete": False}

    async def generate_answer_async(
        self,
        query: str,
        context: str,
        intent: Optional[str] = None,
        images=None,
        history: str = "",
    ) -> dict:
        prompt_context = truncate_context(context) if context.strip() else context
        system, user = self._render(query, prompt_context, intent, history)
        try:
            raw = await self.model_client.chat_async(system=system, user=user, images=images)
            return self._parse_with_valid_references(raw, prompt_context)
        except Exception as exc:
            return self._fallback(context, exc)

    @staticmethod
    def _render(
        query: str, context: str, intent: Optional[str], history: str = ""
    ) -> tuple[str, str]:
        if not context or not context.strip():
            return NO_RESULT_PROMPT.render(query=query)
        return render_hybrid_answer_prompt(
            query=query,
            context=context,
            intent=intent or "unknown",
            history=history,
        )

    @staticmethod
    def _parse_with_valid_references(raw: str, context: str) -> dict:
        """只保留证据中存在且回答正文实际引用的编号。

        兼容半角 ``[n]`` 与全角 ``［n］``；若模型给出了引用但过滤后为空，
        记日志（fail-visible），不静默清空。
        """
        answer = parse_answer_json(raw)
        available = {
            int(value)
            for value in re.findall(r"(?m)^[\[\［](\d+)[\]\］]", context or "")
        }
        cited = {
            int(value)
            for value in re.findall(r"[\[\［](\d+)[\]\］]", str(answer.get("answer", "")))
        }
        raw_refs = [int(value) for value in answer.get("references", [])]
        answer["references"] = [
            value for value in raw_refs
            if value in available and value in cited
        ]
        if raw_refs and not answer["references"]:
            logger.warning(
                "回答引用被过滤为空：raw_refs=%s available=%s cited=%s",
                raw_refs, sorted(available), sorted(cited),
            )
        return answer

    @staticmethod
    def _fallback(context: str, exc: Exception) -> dict:
        logger.warning("RAG 回答生成失败: %s", exc)
        if not context or not context.strip():
            answer = (
                "这个问题现在还没法准确回答。你可以补充具体的药品名称、规格，"
                "或者把想了解的症状说得更具体一些，我再继续帮你看。"
            )
        else:
            answer = "这次回答没有正常生成，可以稍后再试一次。"
        return {"answer": answer, "items": [], "confidence": 0.0, "references": []}
