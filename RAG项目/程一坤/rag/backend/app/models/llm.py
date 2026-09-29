"""大模型适配器：OpenAI 兼容接口的统一入口。

项目里所有"调用大模型生成回答"的地方都走这里，好处是：
换模型（qwen-plus / deepseek-chat / 其他兼容接口）只需要改 .env，不动业务代码。

错误处理原则：任何异常都不许静默吞掉，一律转成 LlmApiError 抛给上层，
避免"模型没回答成功，但上层以为成功了"。
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from app.models.http_retry import chat_completions_endpoint

# 传输函数签名：地址、请求头、请求体字节、超时 → 解析后的 JSON 字典
# 单独抽出来是为了测试时可以注入替身，不真实联网
Transport = Callable[[str, dict[str, str], bytes, float], dict]

# 流式传输函数签名：同上，但返回逐行迭代的 SSE 响应体（不解析 JSON）
StreamTransport = Callable[[str, dict[str, str], bytes, float], Any]


class LlmApiError(RuntimeError):
    """大模型调用失败（超时、HTTP 错误、返回结构无效、回答为空）。"""


class OpenAiCompatibleChatClient:
    """OpenAI 兼容的对话客户端。"""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.transport = transport or self._request
        self.stream_transport = stream_transport or self._stream_request

    @property
    def chat_endpoint(self) -> str:
        """对话接口地址；容忍配置里带或不带结尾斜杠。"""
        return chat_completions_endpoint(self.api_base_url)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """发送一次对话请求，返回模型回答正文。

        参数：
        - system_prompt：系统提示词（角色、规则、法源清单等）
        - user_prompt：用户本轮问题（或组装后的完整提问）

        返回：
        - 回答正文（已去除首尾空白，保证非空）
        """
        if not self.api_base_url or not self.api_key or not self.model:
            # 缺少配置时立刻报错，而不是发出一个必然失败的请求
            raise LlmApiError("大模型配置不完整，请检查 LLM_API_BASE_URL / LLM_API_KEY / LLM_MODEL")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.transport(self.chat_endpoint, headers, payload, self.timeout)
        except TimeoutError as error:
            raise LlmApiError("大模型请求超时") from error
        except LlmApiError:
            raise
        except Exception as error:
            # 不把原始异常文本直接透出，避免其中夹带请求头（含密钥）
            raise LlmApiError(f"大模型请求失败：{type(error).__name__}") from error

        return self._extract_answer(response)

    def stream_chat(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """流式对话：逐块产出增量文本（OpenAI 兼容 stream:true SSE 协议）。

        与 chat() 的区别：不在服务端攒完整回答，边生成边产出，
        让 API 层能把 token 实时推给前端（真流式，不是事后切帧）。
        """
        if not self.api_base_url or not self.api_key or not self.model:
            raise LlmApiError("大模型配置不完整，请检查 LLM_API_BASE_URL / LLM_API_KEY / LLM_MODEL")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.stream_transport(self.chat_endpoint, headers, payload, self.timeout)
        except LlmApiError:
            raise
        except Exception as error:
            # 不透出原始异常文本，避免夹带请求头（含密钥）
            raise LlmApiError(f"大模型流式请求失败：{type(error).__name__}") from error

        return self._iter_stream_deltas(response)

    @staticmethod
    def _iter_stream_deltas(response: Any) -> Iterator[str]:
        """解析流式响应体：data: {...delta...} 行产出增量，[DONE] 结束。

        脏行（非 JSON、无 delta.content）跳过不中断，服务端偶发噪音不该炸整个流。
        """
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip() if isinstance(raw_line, bytes) else raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta") or {}
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                content = delta.get("content")
                if content:
                    yield content
        except LlmApiError:
            raise
        except Exception as error:
            raise LlmApiError(f"大模型流式读取中断：{type(error).__name__}") from error

    @staticmethod
    def _extract_answer(response: dict) -> str:
        """从返回结构中取出回答正文；结构不对或内容为空都视为失败。"""
        try:
            choices = response["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LlmApiError("大模型返回结构无效") from error

        if not isinstance(content, str) or not content.strip():
            # 推理模型（如 DeepSeek 思考模式）会先输出思维链，
            # 若 max_tokens 被思考过程占满，正文就会是空的 —— 报错要说清这一点，
            # 否则会被误判成"模型故障"，排查半天
            reason = "可能是推理过程占满了 max_tokens 额度，请提高 LLM_MAX_TOKENS"
            raise LlmApiError(f"大模型返回了空回答（{reason}）")
        return content.strip()

    def _request(
        self,
        url: str,
        headers: dict[str, str],
        payload: bytes,
        timeout: float,
    ) -> dict:
        """真实 HTTP 调用；错误信息里只保留状态码，不回显请求内容。"""
        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise LlmApiError(f"大模型 HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise LlmApiError(f"大模型网络请求失败：{error.reason}") from error
        except TimeoutError:
            raise
        except json.JSONDecodeError as error:
            raise LlmApiError("大模型返回 JSON 无效") from error

    def _stream_request(
        self,
        url: str,
        headers: dict[str, str],
        payload: bytes,
        timeout: float,
    ) -> Any:
        """真实流式 HTTP 调用；返回可逐行迭代的响应体（行含末尾换行符）。"""
        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
            return response
        except urllib.error.HTTPError as error:
            raise LlmApiError(f"大模型 HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise LlmApiError(f"大模型网络请求失败：{error.reason}") from error
        except TimeoutError:
            raise LlmApiError("大模型请求超时")


def build_chat_client_from_settings() -> OpenAiCompatibleChatClient:
    """按应用配置创建对话客户端。

    放在这里而不是让调用方各自组装，是为了让"模型从哪来"只有一个出处。
    """
    from app.core.config import settings

    return OpenAiCompatibleChatClient(
        api_base_url=settings.llm_api_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )
