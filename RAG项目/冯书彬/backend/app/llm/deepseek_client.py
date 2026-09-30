import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from backend.app.llm.base import LlmRequest
from backend.app.services.model_log_service import ModelCallMetadata, record_model_call

logger = logging.getLogger(__name__)

_STREAM_DONE = "__DEEPSEEK_STREAM_DONE__"


class DeepSeekUnavailableError(RuntimeError):
    # DeepSeek 不可用时统一抛出受控异常，避免上游展示半截未校验回答。
    pass


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat", timeout_seconds: float = 30.0):
        # API Key 只存于客户端内存，任何日志和异常都不得输出该值。
        self._api_key = api_key
        # 去掉尾部斜杠，避免拼接 endpoint 时出现双斜杠。
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def stream_chat(self, request: LlmRequest | dict[str, Any]) -> AsyncIterator[str]:
        # 兼容简报中的 dict 示例，同时内部统一转换为 LlmRequest。
        llm_request = self._coerce_request(request)
        start_time = time.perf_counter()
        token_count = self._estimate_token_count(llm_request.messages)
        error_type: str | None = None
        reason_code = "unknown"
        try:
            # 先在客户端完整接收并验证流，成功后再 yield，避免上游展示半截未校验回答。
            chunks = await self._collect_validated_chunks(llm_request)
            for chunk in chunks:
                yield chunk
        except DeepSeekUnavailableError as exc:
            # 记录异常类别而不是异常详情，防止响应体中夹带敏感内容。
            error_type = exc.__class__.__name__
            reason_code = "controlled_unavailable"
            self._log_unavailable_warning(llm_request, error_type, reason_code)
            raise
        except (httpx.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # 网络、超时、HTTP 和流格式错误统一转为受控异常。
            error_type = exc.__class__.__name__
            reason_code = "transport_or_stream_error"
            self._log_unavailable_warning(llm_request, error_type, reason_code)
            raise DeepSeekUnavailableError("DeepSeek 服务暂不可用") from exc
        finally:
            # finally 在成功和失败路径都会执行，确保模型调用元数据完整落盘。
            status = "failure" if error_type else "success"
            self._record_metadata(llm_request, token_count, start_time, status, error_type, reason_code)

    async def _collect_validated_chunks(self, request: LlmRequest) -> list[str]:
        # payload 保持 OpenAI 兼容格式，不在日志中输出完整 messages。
        payload = {"model": self._model, "messages": request.messages, "stream": True}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        timeout = httpx.Timeout(self._timeout_seconds)
        chunks: list[str] = []
        stream_done = False
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self._base_url}/chat/completions", json=payload, headers=headers) as response:
                # 先确认请求被服务端接受，失败时不输出任何模型内容。
                response.raise_for_status()
                logger.info("DeepSeek 流式请求已接受", extra={"trace_id": request.trace_id, "model": self._model})
                async for line in response.aiter_lines():
                    chunk = self._parse_stream_line(line)
                    if chunk == _STREAM_DONE:
                        stream_done = True
                        break
                    if chunk is None:
                        continue
                    chunks.append(chunk)
        if not stream_done:
            raise DeepSeekUnavailableError("DeepSeek 流未正常结束")
        return chunks

    def _parse_stream_line(self, line: str) -> str | Literal["__DEEPSEEK_STREAM_DONE__"] | None:
        # 空行和注释行不包含模型增量内容。
        if not line or line.startswith(":"):
            return None
        # DeepSeek/OpenAI 兼容 SSE 使用 data: 前缀。
        if not line.startswith("data: "):
            raise DeepSeekUnavailableError("DeepSeek 流格式异常")
        data = line.removeprefix("data: ").strip()
        # [DONE] 表示服务端正常结束流。
        if data == "[DONE]":
            return _STREAM_DONE
        payload = json.loads(data)
        choices = payload["choices"]
        if not choices:
            return None
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content is None:
            return None
        if not isinstance(content, str):
            raise DeepSeekUnavailableError("DeepSeek 内容格式异常")
        return content

    def _log_unavailable_warning(self, request: LlmRequest, error_type: str, reason_code: str) -> None:
        # warning 只记录短代码和元数据，不记录 Prompt、回答、上下文、响应体或 API Key。
        logger.warning(
            "DeepSeek 调用不可用",
            extra={
                "provider": "deepseek",
                "model": self._model,
                "trace_id": request.trace_id,
                "error_type": error_type,
                "reason_code": reason_code,
                "prompt_version": request.prompt_version,
                "knowledge_base_version": request.knowledge_base_version,
            },
        )

    def _record_metadata(
        self,
        request: LlmRequest,
        token_count: int,
        start_time: float,
        status: str,
        error_type: str | None,
        reason_code: str,
    ) -> None:
        # 耗时只记录毫秒数，便于排查超时和性能问题。
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        # 是否使用案例引用只看引用类型，不记录引用全文。
        case_citation_used = any(citation.kind == "案例参考" for citation in request.citations)
        metadata = ModelCallMetadata(
            provider="deepseek",
            model=self._model,
            model_version=request.model_version or self._model,
            prompt_version=request.prompt_version,
            knowledge_base_version=request.knowledge_base_version,
            token_count=token_count,
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            trace_id=request.trace_id,
            redaction_applied=True,
            memory_used=request.memory_used,
            case_citation_used=case_citation_used,
            citation_validation_passed=request.citation_validation_passed,
            reason_code=reason_code,
        )
        record_model_call(metadata)

    def _coerce_request(self, request: LlmRequest | dict[str, Any]) -> LlmRequest:
        # dict 入口仅用于测试和兼容，未提供字段时使用安全默认值。
        if isinstance(request, LlmRequest):
            return request
        return LlmRequest(
            messages=list(request.get("messages", [])),
            citations=list(request.get("citations", [])),
            trace_id=str(request.get("trace_id", "")),
            prompt_version=str(request.get("prompt_version", "unknown")),
            knowledge_base_version=str(request.get("knowledge_base_version", "unknown")),
            model_version=request.get("model_version"),
            memory_used=bool(request.get("memory_used", False)),
            citation_validation_passed=bool(request.get("citation_validation_passed", False)),
        )

    def _estimate_token_count(self, messages: list[dict[str, Any]]) -> int:
        # MVP 无 tokenizer 时按字符粗略估算，只用于元数据趋势分析。
        char_count = sum(len(str(message.get("content", ""))) for message in messages)
        return max(1, char_count // 4) if char_count else 0
