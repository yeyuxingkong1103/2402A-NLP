"""外部 HTTP 调用的共用「请求 + 指数退避重试」工具（批次 22 新增）。

为什么要有这个文件：
批次 22 新增 MinerU / Qwen-VL 两个适配器，它们需要的重试语义与既有的
`app/models/embedding.py` 完全一致 —— 超时 / HTTP 429 / HTTP 5xx 指数退避重试，
400/401/403 等其余状态码立即抛出（不浪费重试次数与调用额度）。

这里**不是**把 embedding.py 重构过来，而是把同一套语义抽出来给新代码复用。
原因：embedding.py 的重试行为被既有测试逐字锁定（每次重试的日志文案、异常消息、
预算判定都参与断言），重构它属于"没有收益的风险"。所以 embedding.py 保持原样，
本文件是它的"同语义副本"，供新适配器使用。

错误分类（与 embedding.py 一致）：
- 可重试：`TimeoutError`、HTTP 429、HTTP 5xx
- 不可重试：HTTP 400/401/403/404 等，以及 JSON/结构类错误
"""

from __future__ import annotations

import http.client
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class HttpApiError(RuntimeError):
    """HTTP 调用或响应校验失败。

    status_code：HTTP 状态码（非 HTTP 错误为 None）。重试分类依赖它：
    429 / 5xx 可重试，400/401/403 等其余状态码不可重试。
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def chat_completions_endpoint(api_base_url: str) -> str:
    """对话接口地址；容忍配置里带或不带结尾斜杠。

    llm.py 与 qwen_vl.py 的 `chat_endpoint` 共用此实现（逐字相同的 3 行
    原本在两处各抄一份，去重时行为必须保持不变）。
    """
    return f"{api_base_url.rstrip('/')}/chat/completions"


def is_retryable(error: Exception) -> bool:
    """可重试错误分类：连接/读超时、HTTP 429、HTTP 5xx；其余一律不重试。"""
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, HttpApiError):
        code = error.status_code
        return code == 429 or (code is not None and 500 <= code <= 599)
    return False


def call_with_retry(
    func: Callable[[], Any],
    *,
    attempts: int,
    backoff_seconds: float,
    total_budget_seconds: float,
    what: str,
    error_type: type[Exception] = HttpApiError,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Any:
    """执行 func，失败为可重试错误时按指数退避重试。

    参数：
    - attempts：**额外**重试次数（总尝试次数 = 1 + attempts），与
      `EMBEDDING_RETRY_ATTEMPTS` 的语义保持一致。
    - backoff_seconds：首次退避时长，第 n 次重试睡 `backoff * 2**n`。
    - total_budget_seconds：重试总耗时预算；下一次退避会突破预算时立即放弃，
      不再浪费一次注定失败的调用。
    - what：出现在日志与错误消息里的动作名（如 "MinerU 提交任务"），
      便于排查时区分是哪一个请求失败。
    - sleep / monotonic：可注入，测试用它跳过真实等待。
    """
    request_id = uuid.uuid4().hex[:12]
    started = monotonic()
    max_attempts = 1 + max(0, attempts)

    for attempt in range(max_attempts):
        try:
            result = func()
            if attempt:
                logger.info(
                    "%s 重试成功 request_id=%s 第 %d 次尝试",
                    what,
                    request_id,
                    attempt + 1,
                )
            return result
        except TimeoutError as error:
            last_error: Exception = error
        except HttpApiError as error:
            if not is_retryable(error):
                # 400/401/403 等：立即抛出，不浪费重试与额度
                raise
            last_error = error

        # 走到这里说明本次失败为可重试错误
        if attempt + 1 >= max_attempts:
            raise error_type(
                f"{what}请求失败（已重试 {attempt} 次后放弃，"
                f"错误类型 {type(last_error).__name__}）：{last_error}"
            ) from last_error

        backoff = backoff_seconds * (2**attempt)
        if monotonic() - started + backoff > total_budget_seconds:
            raise error_type(
                f"{what}请求失败（重试总耗时预算 {total_budget_seconds}s 已耗尽，"
                f"错误类型 {type(last_error).__name__}）：{last_error}"
            ) from last_error

        logger.warning(
            "%s 请求失败，准备重试 request_id=%s 第 %d/%d 次（错误类型：%s）",
            what,
            request_id,
            attempt + 1,
            attempts,
            type(last_error).__name__,
        )
        sleep(backoff)

    raise error_type(f"{what}请求失败")  # 理论不可达：循环内必 return 或 raise


def request_bytes(
    url: str,
    headers: dict[str, str],
    payload: bytes,
    timeout: float,
    *,
    method: str = "POST",
) -> bytes:
    """发送请求并返回原始响应体（供 MinerU 的 JSON 接口使用）。

    与 `request_json` 共享同一套错误分类：HTTPError 带状态码抛出，
    URLError 里的超时统一转成可重试的 TimeoutError。

    注意：空 payload 会转成 `data=None`。urllib 只要看到 `data is not None`
    就会自动补 `Content-Type: application/x-www-form-urlencoded`，还会带上
    `Content-Length: 0`；对 GET 轮询接口来说这是多余且可能被拒的头。
    """
    request = urllib.request.Request(
        url,
        data=payload if payload else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        # 保留状态码供重试分类（429/5xx 可重试）
        raise HttpApiError(f"HTTP {error.code} {url}", status_code=error.code) from error
    except urllib.error.URLError as error:
        reason = error.reason
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise TimeoutError(str(reason)) from error
        raise HttpApiError(f"网络请求失败：{reason}") from error
    except TimeoutError:
        raise


def request_raw(
    url: str,
    *,
    method: str,
    payload: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> bytes:
    """用 `http.client` 发请求，**只发调用方给的头，不做任何自动补充**。

    为什么不能用 urllib：对象存储（OSS/COS 等）的预签名 URL 是按
    "空 Content-Type" 参与签名计算的，而 urllib 只要带 data 就会自动补
    `Content-Type: application/x-www-form-urlencoded` —— 多这一个头就会
    被判定为 `SignatureDoesNotMatch` 而 403。

    批次 22 实测（MinerU 预签名上传地址，同一 URL 形态）：
    | 形态 | 结果 |
    |---|---|
    | curl -T 文件（无 Content-Type） | 200 OK |
    | urllib 无显式头（自动补 Content-Type） | 403 SignatureDoesNotMatch |
    | urllib 显式 Content-Type: application/pdf | 403 SignatureDoesNotMatch |
    | http.client 不发 Content-Type | 200 OK |

    结论：预签名 URL 的请求必须精确控制请求头，故这里用 http.client。
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise HttpApiError(f"不支持的 URL 协议：{parts.scheme or '(空)'}")
    target = parts.path + (f"?{parts.query}" if parts.query else "")

    connection_class = (
        http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_class(parts.hostname, parts.port, timeout=timeout)
    try:
        connection.request(method, target, body=payload, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        status = response.status
    except TimeoutError:
        raise
    except (http.client.HTTPException, OSError) as error:
        raise HttpApiError(f"{method} 请求失败（{type(error).__name__}）") from error
    finally:
        connection.close()

    if status >= 400:
        raise HttpApiError(
            f"HTTP {status} {parts.netloc}", status_code=status
        )
    return body


def request_json(
    url: str,
    headers: dict[str, str],
    payload: bytes,
    timeout: float,
    *,
    method: str = "POST",
) -> dict:
    """发送请求并把响应体解析成 JSON 字典。

    解析失败属于不可重试错误：重试同样的请求只会拿到同样的坏响应。
    """
    body = request_bytes(url, headers, payload, timeout, method=method)
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HttpApiError(f"返回 JSON 无效：{url}") from error
    if not isinstance(decoded, dict):
        raise HttpApiError(f"返回结构不是 JSON 对象：{url}")
    return decoded


# 传输函数签名：地址、请求头、请求体字节、超时 → 解析后的 JSON 字典。
# 与 embedding.py / llm.py 的同名类型保持一致，便于替换与阅读。
Transport = Callable[[str, dict[str, str], bytes, float], dict]

# 字节传输函数签名：同上，但返回原始响应体（上传 / 下载用）
BytesTransport = Callable[[str, dict[str, str], bytes, float], bytes]
