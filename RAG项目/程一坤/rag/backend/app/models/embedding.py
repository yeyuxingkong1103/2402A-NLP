import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


class EmbeddingApiError(RuntimeError):
    """Embedding 服务调用或响应校验失败。

    status_code：HTTP 状态码（非 HTTP 错误为 None），重试分类依赖它：
    429 / 5xx 可重试，400/401/403 等其余状态码不可重试。
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _is_retryable(error: Exception) -> bool:
    """可重试错误分类：连接/读超时、HTTP 429、HTTP 5xx；其余一律不重试。"""
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, EmbeddingApiError):
        code = error.status_code
        return code == 429 or (code is not None and 500 <= code <= 599)
    return False


Transport = Callable[[str, dict[str, str], bytes, float], dict]


class SiliconFlowEmbeddingClient:
    """调用硅基流动 OpenAI 兼容 Embeddings 接口。

    批次 15：内置可重试错误（超时 / 429 / 5xx）的指数退避重试；
    400/401/403 等不可重试错误立即抛出。重试参数可经 .env 配置：
    EMBEDDING_RETRY_ATTEMPTS / EMBEDDING_RETRY_BACKOFF_SECONDS /
    EMBEDDING_RETRY_TOTAL_BUDGET_SECONDS。
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        dimension: int,
        timeout: float = 60.0,
        transport: Transport | None = None,
        retry_attempts: int | None = None,
        retry_backoff_seconds: float | None = None,
        retry_total_budget_seconds: float | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        self.transport = transport or self._request
        self.retry_attempts = (
            int(os.getenv("EMBEDDING_RETRY_ATTEMPTS", "3"))
            if retry_attempts is None
            else retry_attempts
        )
        self.retry_backoff_seconds = (
            float(os.getenv("EMBEDDING_RETRY_BACKOFF_SECONDS", "0.5"))
            if retry_backoff_seconds is None
            else retry_backoff_seconds
        )
        self.retry_total_budget_seconds = (
            float(os.getenv("EMBEDDING_RETRY_TOTAL_BUDGET_SECONDS", "15"))
            if retry_total_budget_seconds is None
            else retry_total_budget_seconds
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingApiError("Embedding 输入文本不能为空")

        payload = json.dumps(
            {
                "model": self.model,
                "input": list(texts),
                "encoding_format": "float",
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._embed_with_retry(headers, payload, len(texts))

        items = response.get("data") if isinstance(response, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbeddingApiError("Embedding 返回数量与输入数量不一致")

        try:
            ordered_items = sorted(items, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in ordered_items]
        except (KeyError, TypeError) as error:
            raise EmbeddingApiError("Embedding 返回格式无效") from error

        if any(
            not isinstance(vector, list) or len(vector) != self.dimension
            for vector in vectors
        ):
            raise EmbeddingApiError(
                f"Embedding 向量维度错误，要求 {self.dimension} 维"
            )
        return vectors

    def _embed_with_retry(self, headers: dict[str, str], payload: bytes, count: int) -> dict:
        """带指数退避重试的请求；不可重试错误立即抛出，不消耗重试。"""
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        max_attempts = 1 + max(0, self.retry_attempts)
        for attempt in range(max_attempts):
            try:
                response = self.transport(self.api_url, headers, payload, self.timeout)
                if attempt:
                    logger.info(
                        "Embedding 重试成功 request_id=%s 第 %d 次尝试",
                        request_id,
                        attempt + 1,
                    )
                return response
            except TimeoutError as error:
                last_error: Exception = error
            except EmbeddingApiError as error:
                if not _is_retryable(error):
                    # 400/401/403 等：立即抛出，不浪费重试与额度
                    raise
                last_error = error
            # 走到这里说明本次失败为可重试错误
            if attempt + 1 >= max_attempts:
                raise EmbeddingApiError(
                    f"Embedding 请求失败（已重试 {attempt} 次后放弃，"
                    f"错误类型 {type(last_error).__name__}）：{last_error}"
                ) from last_error
            backoff = self.retry_backoff_seconds * (2**attempt)
            if time.monotonic() - started + backoff > self.retry_total_budget_seconds:
                raise EmbeddingApiError(
                    f"Embedding 请求失败（重试总耗时预算 "
                    f"{self.retry_total_budget_seconds}s 已耗尽，错误类型 "
                    f"{type(last_error).__name__}）：{last_error}"
                ) from last_error
            logger.warning(
                "Embedding 请求失败，准备重试 request_id=%s 第 %d/%d 次（错误类型：%s）",
                request_id,
                attempt + 1,
                self.retry_attempts,
                type(last_error).__name__,
            )
            time.sleep(backoff)
        raise EmbeddingApiError("Embedding 请求失败")  # 理论不可达：循环内必 return 或 raise

    def _request(
        self,
        url: str,
        headers: dict[str, str],
        payload: bytes,
        timeout: float,
    ) -> dict:
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
            # 保留状态码供重试分类（429/5xx 可重试）
            raise EmbeddingApiError(
                f"Embedding HTTP {error.code}", status_code=error.code
            ) from error
        except urllib.error.URLError as error:
            reason = error.reason
            # 连接/读超时统一转成 TimeoutError（可重试）；其余网络错误不可重试
            if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
                raise TimeoutError(str(reason)) from error
            raise EmbeddingApiError(f"Embedding 网络请求失败：{reason}") from error
        except TimeoutError:
            raise
        except json.JSONDecodeError as error:
            raise EmbeddingApiError("Embedding 返回 JSON 无效") from error
