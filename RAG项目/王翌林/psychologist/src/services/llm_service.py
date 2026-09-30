"""大模型调用：DeepSeek 在线 API（OpenAI 兼容接口）。

职责：把"和模型对话"收敛到一处，对上层屏蔽 SDK 细节与异常类型。
上游（rag_service / retrieval_service / memory_service / eval_service）只调用本模块，
不直接 import openai——这样换模型厂商、改超时、加重试都只需改这一个文件。

被谁调用：rag_service（正式对话，流式/非流式）、retrieval_service（Query 改写）、
memory_service（会话摘要）、eval_service（LLM-as-Judge 打分）。

容错设计（两种失败策略）：
- chat / chat_stream：直接面向用户，失败抛 ExternalServiceError，由上层决定是否提示；
- simple_complete：辅助能力（改写/摘要/起标题/打分），失败返回空串而非抛错——
  辅助步骤失败不该让用户整段对话失败，上层再用规则兜底。
"""
import time
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI, OpenAIError

from src.core.config import settings
from src.core.exceptions import ExternalServiceError
from src.core.logging import get_logger

logger = get_logger("llm")

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """惰性单例创建 OpenAI 客户端：首次调用才建连，避免 import 期就因缺 key 崩溃。
    未配置 LLM_API_KEY 时立刻抛明确异常——属于部署配置问题，越早暴露越好。
    """
    global _client
    if _client is None:
        if not settings.llm_api_key:
            raise ExternalServiceError("未配置 LLM_API_KEY，请在 .env 中设置")
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            max_retries=2,
        )
        logger.info("LLM 客户端已创建：%s model=%s", settings.llm_base_url, settings.llm_model)
    return _client


def _extra_body() -> Optional[Dict[str, Any]]:
    """DeepSeek 思考开关：deepseek-flash 默认先输出 reasoning_content（content 为空），
    陪伴对话场景关闭思考（LLM_THINKING=disabled）以保证正文直出、低延迟。"""
    if settings.llm_thinking == "disabled":
        return {"thinking": {"type": "disabled"}}
    if settings.llm_thinking == "enabled":
        return {"thinking": {"type": "enabled"}}
    return None


def chat(messages: List[Dict[str, str]], temperature: Optional[float] = None,
         max_tokens: Optional[int] = None, model: Optional[str] = None,
         timeout: Optional[float] = None) -> Dict[str, Any]:
    """非流式调用，返回 {content, tokens, finish_reason}。timeout 为单请求级超时（秒）。"""
    temperature = settings.llm_temperature if temperature is None else temperature
    max_tokens = max_tokens or settings.llm_max_tokens
    model = model or settings.llm_model
    start = time.time()
    try:
        resp = get_client().chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=False, extra_body=_extra_body(),
            timeout=timeout,
        )
    except OpenAIError as exc:
        # 统一转成业务异常类型：上层（API）只需处理 ExternalServiceError，无需认识 openai 的异常体系
        logger.error("LLM 调用失败：%s", exc)
        raise ExternalServiceError(f"大模型调用失败：{exc}") from exc

    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    result = {
        "content": (choice.message.content or "").strip(),
        "tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "finish_reason": choice.finish_reason or "stop",
        "elapsed_ms": int((time.time() - start) * 1000),
    }
    logger.info("LLM 非流式完成 tokens=%s 耗时=%sms", result["tokens"], result["elapsed_ms"])
    return result


def chat_stream(messages: List[Dict[str, str]], temperature: Optional[float] = None,
                max_tokens: Optional[int] = None, model: Optional[str] = None,
                timeout: Optional[float] = None
                ) -> Generator[str, None, None]:
    """流式调用，逐段产出文本增量。"""
    temperature = settings.llm_temperature if temperature is None else temperature
    max_tokens = max_tokens or settings.llm_max_tokens
    model = model or settings.llm_model
    start = time.time()
    try:
        stream = get_client().chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=True, extra_body=_extra_body(),
            timeout=timeout,
        )
        for chunk in stream:
            # 部分 chunk 只带 usage/元信息而没有 choices，跳过以免索引越界
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece
    except OpenAIError as exc:
        logger.error("LLM 流式调用失败：%s", exc)
        raise ExternalServiceError(f"大模型流式调用失败：{exc}") from exc
    finally:
        logger.info("LLM 流式完成 耗时=%sms", int((time.time() - start) * 1000))


def simple_complete(prompt: str, system: str = "", temperature: float = 0.3,
                    max_tokens: int = 512, timeout: Optional[float] = None) -> str:
    """单轮辅助调用（Query 改写、摘要、起标题）。失败时返回空字符串，不阻断主流程。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        return chat(messages, temperature=temperature, max_tokens=max_tokens,
                    timeout=timeout)["content"]
    except Exception as exc:
        # 关键降级点：辅助调用失败返回空串，由调用方决定用规则兜底（如摘要）还是跳过（如改写）
        logger.warning("辅助 LLM 调用失败（降级）：%s", exc)
        return ""