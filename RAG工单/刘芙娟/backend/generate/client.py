"""LLM 出网调用。**本包唯一出网的地方。**

    stream_model(...)  —— 异步流式（服务端用，逐段吐给 SSE）
    call_model(...)    —— 同步/异步非流式（CLI 与脚本用，返回完整文本）

两者都走 OpenAI 兼容的 `/chat/completions`，都带重试。

---

## 密钥

`AGICTO_API_KEY` 只从环境变量/`.env` 读取（constitution 原则 III）。本模块
MUST NOT 提供任何默认值兜底 —— 缺密钥时报错并指出该填哪个文件，
MUST NOT 以空密钥发请求：那会拿到一个 401，然后被重试逻辑掩盖成"网络问题"，
把一个配置错误伪装成偶发故障。

## 为什么用 `httpx.AsyncClient` 而不是 `httpx.Client`

生成要 5–10 秒。用同步客户端会把这 5–10 秒**整个事件循环**堵住 —— 期间其它请求
一个都进不来。这比 `backend/retrieve/service.py` 里那 50 ms 的阻塞严重两个数量级，
所以在这一处破例用异步（R12 记录的"暂不改同步路径"不覆盖这里）。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from . import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_S,
    MAX_RETRIES,
    MIN_RETRY_BUDGET_S,
)

__all__ = ["LLMError", "stream_model", "call_model", "API_KEY_ENV", "DEFAULT_MODEL"]

# 流结束哨兵。OpenAI 兼容接口在最后一个事件前发它。
_STREAM_DONE = "[DONE]"


class LLMError(Exception):
    """调用失败：缺密钥、超时、重试耗尽、响应无法解析。

    ⚠️ 刻意**不继承** `ValueError` —— 与 `QueryError` / `RetrievalError` /
    `PromptError` 同一取向：继承内建异常会让 `except ValueError` 意外捕获到它。
    """

    def __init__(self, message: str, *, attempts: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.attempts = attempts


class _AttemptFailed(Exception):
    """单次尝试失败，可重试。**内部异常，不对外暴露。**"""


def _check_key(api_key: str | None) -> None:
    if not api_key:
        raise LLMError(
            "缺少 %s。请在仓库根创建 .env（可复制 .env.example）并填入真实密钥：\n"
            "    %s=<你的密钥>\n"
            "  该变量 MUST NOT 硬编码进代码（constitution 原则 III）。"
            % (API_KEY_ENV, API_KEY_ENV)
        )


def _payload(model: str, prompt: str, *, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # 医疗问答要的是可复现与忠于原文，不是文采。
        # 0.2 而非 0：完全贪心在长中文输出上容易陷入重复循环。
        "temperature": 0.2,
        "stream": stream,
    }


def _headers(api_key: str) -> dict:
    return {"Authorization": "Bearer " + api_key}


def _brief(response: httpx.Response) -> str:
    """错误摘要。**不回显请求头** —— 那里面有 Authorization。"""

    return "%d %s" % (response.status_code, response.text[:200].replace("\n", " "))


# ------------------------------------------------------------------ 流式

async def _iter_once(
    url: str, payload: dict, api_key: str, timeout: float
) -> AsyncIterator[str]:
    """单次流式请求，逐段产出文本增量。

    流在收到 `[DONE]` 之前结束 → 抛 `_AttemptFailed`。
    这一条是必需的：SSE 流被中途掐断时，HTTP 层是**正常结束**的（连接关闭），
    不检查哨兵就会把半截回答当成完整回答交给用户，而用户没有任何办法看出
    它少了后半段。
    """

    saw_done = False
    emitted = False
    # 非 `data:` 行的前几条。**这是给"密钥写错"这个场景准备的。**
    #
    # 中转站在认证失败时不一定会给 4xx —— 实测 agicto 对无效密钥返回的是
    # **HTTP 200 + 一段 JSON 错误体**。那样状态码检查放行、SSE 解析器又一行都
    # 认不出来，最终报的是"流未正常结束"，而使用者真正需要看到的是"密钥无效"。
    # 把原始响应体的开头几行带进错误信息，这一类问题就从"无从下手"变成
    # "一眼看出该改哪里"。
    sniffer: list[str] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, json=payload, headers=_headers(api_key)
        ) as response:
            if response.status_code != 200:
                await response.aread()
                # 4xx 是请求本身的问题（密钥错、模型名错），重试只是浪费预算。
                if response.status_code < 500:
                    raise LLMError("接口返回 %s" % _brief(response))
                raise _AttemptFailed("接口返回 %s" % _brief(response))

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    if len(sniffer) < 3 and line.strip():
                        sniffer.append(line.strip()[:120])
                    continue
                data = line[5:].strip()
                if data == _STREAM_DONE:
                    saw_done = True
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (ValueError, KeyError, IndexError, TypeError):
                    # 单个事件解析不了不该中断整条流（think 模型会发空 delta、
                    # 有的中转站会插额外事件）。跳过它，让哨兵检查兜底。
                    continue
                if delta:
                    emitted = True
                    yield delta

    if not saw_done:
        detail = ""
        if not emitted and sniffer:
            detail = "；响应体开头：" + " / ".join(sniffer)
        raise _AttemptFailed("流未正常结束（未收到 %s）%s" % (_STREAM_DONE, detail))


async def stream_model(
    prompt: str,
    *,
    api_key: str | None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = MAX_RETRIES,
) -> AsyncIterator[str]:
    """异步流式调用。

    ⚠️ **重试只在第一个增量吐出去之前有效。**

    第一个增量一旦离开本函数，调用方就可能已经把它发给了用户 —— 此后重试意味着
    用户会看到"前半段 + 重新开始的后半段"，比半截回答更糟。所以：首段之前失败
    → 照常重试；首段之后失败 → 直接抛 `LLMError`，由调用方按"回答中断"处理。

    ⚠️ **总预算约束**：`docs/05` §2.4 给整个回答的延迟预算是 10 s。若三次尝试各给
    10 s，最坏会到 30 s —— 用户早已放弃，服务端还在重试。因此维护一个总预算
    deadline，每次尝试只拿到剩余时间。
    """

    _check_key(api_key)
    url = base_url.rstrip("/") + "/chat/completions"
    payload = _payload(model, prompt, stream=True)

    deadline = time.monotonic() + timeout
    last_error = ""
    started = False

    for attempt in range(max_retries + 1):
        budget = deadline - time.monotonic()
        if started:
            break
        if attempt > 0 and budget < MIN_RETRY_BUDGET_S:
            last_error = "%s（剩余预算 %.1f s，放弃重试）" % (last_error, max(budget, 0.0))
            break

        try:
            async for delta in _iter_once(url, payload, api_key, max(budget, MIN_RETRY_BUDGET_S)):
                started = True
                yield delta
            return
        except LLMError:
            raise
        except _AttemptFailed as exc:
            last_error = str(exc)
        except httpx.HTTPError as exc:
            last_error = "%s: %s" % (type(exc).__name__, exc)

    if started:
        raise LLMError(
            "回答中途中断（已收到部分内容，不再重试以免用户看到重复片段）",
            attempts=attempt + 1,
        )
    raise LLMError(
        "调用模型失败，已尝试 %d 次：%s" % (attempt + 1, last_error), attempts=attempt + 1
    )


# ------------------------------------------------------------------ 非流式

async def call_model(
    prompt: str,
    *,
    api_key: str | None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = MAX_RETRIES,
) -> str:
    """非流式调用，返回完整文本。

    与 `stream_model` 共用 `_iter_once`，因此**不会**出现"两条路径的成功条件不同"
    这种漂移。区别只在失败语义：这里的增量被本函数攒着、没有第二个人看到，
    所以**中途失败也可以整体重试** —— 把已攒的部分丢掉重来，用户不受影响。
    """

    _check_key(api_key)
    url = base_url.rstrip("/") + "/chat/completions"
    payload = _payload(model, prompt, stream=False)

    deadline = time.monotonic() + timeout
    last_error = ""

    for attempt in range(max_retries + 1):
        budget = deadline - time.monotonic()
        if attempt > 0 and budget < MIN_RETRY_BUDGET_S:
            last_error = "%s（剩余预算 %.1f s，放弃重试）" % (last_error, max(budget, 0.0))
            break

        try:
            async with httpx.AsyncClient(timeout=max(budget, MIN_RETRY_BUDGET_S)) as client:
                response = await client.post(url, json=payload, headers=_headers(api_key))
            if response.status_code != 200:
                if response.status_code < 500:
                    raise LLMError("接口返回 %s" % _brief(response))
                raise _AttemptFailed("接口返回 %s" % _brief(response))

            text = response.json()["choices"][0]["message"]["content"]
            if not text.strip():
                raise _AttemptFailed("响应正文为空")
            return text

        except LLMError:
            raise
        except _AttemptFailed as exc:
            last_error = str(exc)
        except httpx.HTTPError as exc:
            last_error = "%s: %s" % (type(exc).__name__, exc)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            last_error = "响应结构不符：%s" % exc

    raise LLMError(
        "调用模型失败，已尝试 %d 次：%s" % (attempt + 1, last_error), attempts=attempt + 1
    )


def run_sync(coro):
    """在**没有**事件循环的上下文里跑协程（CLI 用）。

    ⚠️ 服务端 MUST NOT 调用它 —— 在已运行的事件循环里 `asyncio.run` 会抛
    `RuntimeError`。服务端直接 `await`。
    """

    return asyncio.run(coro)
