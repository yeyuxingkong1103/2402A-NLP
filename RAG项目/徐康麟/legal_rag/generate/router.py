# -*- coding: utf-8 -*-
"""模型路由层。

三级选路 + 容灾降级
-------------------
选路优先级（高 → 低）::

    任务级(task_providers)  >  角色级(role.default_provider)  >  全局(config.llm_provider)
    >  config.llm_fallbacks  >  默认兜底链

默认兜底链是 ``deepseek(仅当有 DEEPSEEK_API_KEY) → mock``，但**对远端生产端点不再接 mock**
（t120 F1 修的就是这条静默降级）：

* 主线是本地 Ollama（免密钥、免 GPU），它挂了没有任何代价地降级到在线 API；
* 没有 key 时跳过 deepseek，**本地后端**（ollama/local）最终落到 ``mock``——
  离线/演示环境保持既有行为，用户总能拿到一个带检索引用的回答；
* **远端生产端点**（``openai_compat`` / ``vllm`` / ``sglang`` …）失败时**不再默认接 mock**：
  那会让用户拿到 HTTP 200 的假答案（t120 F1）。此时链路直接抛
  :class:`~legal_rag.generate.llm_base.LLMError`（带 ``reason``），由 API 层映射成 503 +
  可读原因 + 可重试；**除非**运维显式写了 ``LLM_FALLBACKS=mock`` —— 那是明确选择，
  响应会带 ``degraded=true`` + ``degraded_reason``，答案里也会加一句可见提示；
* 每一次真实降级都打 **WARNING**（含**可分类原因** ``reason=`` 与截断摘要）并计数
  ``llm_degraded_total{from,to}``。

并发安全
--------
* 客户端缓存加锁（双重检查），避免并发下重复构造；
* 流式调用的状态全部是**调用级局部变量**（``first`` / ``iterator`` / 闭包变量），
  不挂在 router 或 client 上，因此多请求共用同一个 router 也不会串包。
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator

from .. import metrics as M
from ..config import RagConfig
# P0.5：生成侧耗时观测（L1 直方图 + L2 慢调用清单），见 docs/REFACTOR-PLAN.md §3.7
from ..observability import timed
from ..schemas import Role
from .llm_base import LLMClient, LLMError, build_llm_client, failure_reason, summarize_exception
from .llm_metrics import mark_degraded, mark_route_degraded
from .ollama import OLLAMA_PROVIDERS, resolve_ollama_model

logger = logging.getLogger(__name__)

#: 「真实远端/本地服务」后端：只有这些才自动追加默认兜底链。
#: 未知 provider 名（配置写错）保持严格模式，直接报错，不静默吞掉。
REAL_PROVIDERS = frozenset({
    "ollama", "local", "ollama_local", "deepseek", "openai_compat", "openai-compat",
    "openai", "vllm", "sglang", "xinference", "doubao", "siliconflow", "qwen",
})

#: 默认兜底链（真实后端失败时自动接上）
DEFAULT_FALLBACK_CHAIN: tuple[str, ...] = ("deepseek", "mock")

#: 「本地」后端：它们挂了接 mock 是**既定行为**（离线/演示环境，用户本来就知道没有真模型）。
LOCAL_PROVIDERS = frozenset({"ollama", "local", "ollama_local"})

#: mock 是「演示/测试」后端：它只会把问题回显，**没有任何推理**。
#: t120 F1（high）：远端生产端点（openai_compat / vllm / sglang …）失败时若**默认**接上它，
#: 用户会拿到一个 HTTP 200 的假答案，而且从响应里看不出这是编的。因此：
#:
#: * **默认**（没写 ``LLM_FALLBACKS``）⇒ 远端端点失败**不再接 mock**，直接抛
#:   :class:`~legal_rag.generate.llm_base.LLMError`，由 API 层映射成 503 + 可读原因（可重试）；
#: * **显式**写 ``LLM_FALLBACKS=mock`` ⇒ 允许（这是运维的明确选择），但**必须显式标记**：
#:   响应 ``provider=mock``、``degraded=true`` + ``degraded_reason``，且答案里带可见提示。
MOCK_PROVIDER = "mock"

#: 与 deepseek.py 保持一致：只看「有没有」，值绝不进日志
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"


def mock_fallback_allowed(primary: str) -> bool:
    """远端生产端点失败时，是否允许默认兜底到 mock（t120 F1）。"""
    return (primary or "").strip().lower() in LOCAL_PROVIDERS


def deepseek_available() -> bool:
    """是否配置了 DEEPSEEK_API_KEY（只看有无，不读值、不打印）。"""
    return bool((os.environ.get(DEEPSEEK_API_KEY_ENV) or "").strip())


class ModelRouter:
    def __init__(self, config: RagConfig | None = None,
                 clients: dict[str, LLMClient] | None = None,
                 task_providers: dict[str, str] | None = None) -> None:
        self.config = config or RagConfig()
        self.task_providers = dict(task_providers or {})
        self._clients: dict[str, LLMClient] = dict(clients or {})
        self._lock = threading.Lock()
        #: 「远端端点不接 mock」的提示只吼一次（t120 F1）
        self._warned_mock_skip = False

    # ---------- 客户端注册与构建 ----------
    def register(self, provider: str, client: LLMClient) -> None:
        with self._lock:
            self._clients[provider] = client

    def client_for(self, provider: str) -> LLMClient:
        client = self._clients.get(provider)
        if client is not None:
            return client
        with self._lock:                       # 双重检查：并发下只构造一次
            client = self._clients.get(provider)
            if client is None:
                client = self._build(provider)
                self._clients[provider] = client
            return client

    def _model_for(self, provider: str) -> str:
        """Ollama 用 ``config.ollama.llm_model``（默认 qwen2.5:3b），其余用 ``config.llm_model``。"""
        if (provider or "").strip().lower() in OLLAMA_PROVIDERS:
            return resolve_ollama_model(self.config)
        return self.config.llm_model

    def _build(self, provider: str) -> LLMClient:
        ollama = self.config.ollama
        try:
            return build_llm_client(
                provider,
                model=self._model_for(provider),
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                timeout=self.config.llm_timeout,
                deepseek_base_url=self.config.deepseek_base_url,
                openai_base_url=self.config.openai_compat_base_url,
                ollama_base_url=ollama.base_url,
                ollama_api_key=ollama.api_key,
                ollama_top_p=ollama.top_p,
                keep_alive=ollama.keep_alive,
                num_gpu=ollama.num_gpu,
                num_thread=ollama.num_thread,
                max_retries=ollama.max_retries,
                retry_backoff=ollama.retry_backoff,
            )
        except Exception:
            logger.exception("构造大模型客户端失败：provider=%s model=%s",
                             provider, self._model_for(provider))
            raise

    def selector_client(self, *, timeout: float, max_tokens: int = 256) -> LLMClient:
        """给**检索侧小任务**（列表式选择器）用的客户端：主后端、短超时、短输出。

        为什么要单独一个（而不是复用 `client_for`）：
        * **短超时**：选择器是"锦上添花"的一步，大模型端点在抖时绝不能让它把每题都卡满
          `LLM_TIMEOUT`（默认 60 s）—— 那是"速度不能太慢"的直接反面；
        * **短输出**：它只回一行 JSON，`max_tokens` 给 256 足够，省解码时间。

        不缓存：这个客户端只在启动时构造一次（引擎持有），没必要进 `_clients` 复用表
        （那两个用途的 timeout 不同，混用会互相拖累）。
        """
        ollama = self.config.ollama
        provider = str(self.config.llm_provider or "")
        return build_llm_client(
            provider,
            model=self._model_for(provider),
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            deepseek_base_url=self.config.deepseek_base_url,
            openai_base_url=self.config.openai_compat_base_url,
            ollama_base_url=ollama.base_url,
            ollama_api_key=ollama.api_key,
            ollama_top_p=ollama.top_p,
            keep_alive=ollama.keep_alive,
            num_gpu=ollama.num_gpu,
            num_thread=ollama.num_thread,
            max_retries=1,               # 选择器失败就跳过，不做重试放大延迟
            retry_backoff=0.5,
        )

    # ---------- 选路 ----------
    def chain(self, role: Role | None = None, task: str | None = None) -> list[str]:
        """解析出本次请求的候选后端链（高优先级在前，去重）。"""
        chain: list[str] = []

        def push(provider: str | None) -> None:
            name = (provider or "").strip()
            if name and name not in chain:
                chain.append(name)

        if task:
            push(self.task_providers.get(task))
        if role is not None:
            push(role.default_provider)
        push(self.config.llm_provider)
        for provider in self.config.llm_fallbacks:
            push(provider)

        # 真实后端 + 未显式配置兜底链 → 接上默认「deepseek(有 key) → mock」
        primary = (chain[0] if chain else "").lower()
        if not self.config.llm_fallbacks and primary in REAL_PROVIDERS:
            for provider in DEFAULT_FALLBACK_CHAIN:
                if provider == "deepseek" and not deepseek_available():
                    logger.debug("未配置 %s，默认兜底链跳过 deepseek", DEEPSEEK_API_KEY_ENV)
                    continue
                if provider == MOCK_PROVIDER and not mock_fallback_allowed(primary):
                    # t120 F1：远端生产端点失败**不许**静默换成假答案（200 却没有任何推理）。
                    # 真要兜底请显式写 LLM_FALLBACKS=mock —— 那时响应会带 degraded 标记。
                    self._warn_mock_not_allowed(primary)
                    continue
                push(provider)
        return chain

    def _warn_mock_not_allowed(self, primary: str) -> None:
        """同一个 router 实例只吼一次（每次请求都吼会把日志刷爆）。"""
        if self._warned_mock_skip:
            return
        self._warned_mock_skip = True
        logger.warning(
            "[LLM-ROUTE] 主后端 %s 是远端生产端点：默认兜底链**不接 mock** —— 否则它失败时"
            "会变成 HTTP 200 的假答案（t120 F1）。要兜底请显式设 LLM_FALLBACKS=mock"
            "（那时响应 provider=mock、degraded=true，并在答案里给可见提示）。", primary)

    def resolve(self, role: Role | None = None, task: str | None = None) -> LLMClient:
        return self.client_for(self.chain(role, task)[0])

    # ---------- 降级 ----------
    def _degrade(self, provider: str, next_provider: str, exc: BaseException) -> None:
        """记录一次降级：WARNING 日志（含**可分类原因** + 截断摘要）+ 指标 + 上下文标记。"""
        reason = failure_reason(exc)
        summary = summarize_exception(exc)
        if not next_provider:
            logger.error("[LLM-FAIL] 后端 %s 不可用，且已无可用降级后端；reason=%s summary=%s",
                         provider, reason, summary)
            return
        logger.warning("[LLM-FAIL] 后端 %s 不可用，降级到 %s；reason=%s summary=%s",
                       provider, next_provider, reason, summary)
        try:
            M.counter("llm_degraded_total").inc(1, **{"from": provider, "to": next_provider})
        except Exception:  # pragma: no cover - 指标失败不影响降级
            logger.warning("llm_degraded_total 计数失败", exc_info=True)
        mark_degraded(provider, next_provider)
        # 引擎这一层也要知道"降级了、降到了谁、为什么" —— 用来如实填 Answer.degraded
        # 并在答案里放可见提示（t120 F1 的方案②：兜底必须显式标记）。
        mark_route_degraded(provider, next_provider, reason)

    # ---------- 调用（带降级） ----------
    # P0.5：LLM 生成通常是单请求最大的一段耗时；阈值 5000ms ——
    # 大模型本来就慢，低于它的正常调用不必刷日志（但耗时照样进 func_seconds）。
    @timed(operation="router.chat", slow_ms=5000.0)
    def chat(self, messages: list[dict], role: Role | None = None,
             task: str | None = None) -> tuple[str, str]:
        chain = self.chain(role, task)
        errors: list[str] = []
        last: BaseException | None = None
        for index, provider in enumerate(chain):
            try:
                client = self.client_for(provider)
                return client.chat(messages), provider
            except Exception as exc:  # noqa: BLE001 - 逐个降级
                last = exc
                errors.append(f"{provider}: {summarize_exception(exc)}")
                self._degrade(provider, chain[index + 1] if index + 1 < len(chain) else "", exc)
        # 全挂：**不**返回假答案，抛错让上层给出可读原因 + 可重试入口（t120 F1）
        raise LLMError("所有大模型后端都不可用 -> " + " | ".join(errors),
                       reason=failure_reason(last) if last is not None else "unknown")

    # P0.5：``stream`` 只到"探出第一个 chunk"为止（**不含**后续流式输出），
    # 所以它的耗时口径是「首字延迟 TTFT」——阈值给 2000ms。
    # 整轮流式生成的总耗时由上游 ``engine.ask_stream`` 的调用方观测。
    @timed(operation="router.stream_ttft", slow_ms=2000.0)
    def stream(self, messages: list[dict], role: Role | None = None,
               task: str | None = None) -> tuple[str, Iterator[str]]:
        """流式调用：先探出第一个 chunk 以确认后端可用，再交还迭代器。

        返回 ``(实际生效的 provider, 文本迭代器)``。迭代器只捕获本次调用的
        局部变量，天然按请求隔离。
        """
        chain = self.chain(role, task)
        errors: list[str] = []
        last: BaseException | None = None
        for index, provider in enumerate(chain):
            try:
                client = self.client_for(provider)
                iterator = client.stream(messages)
                first = next(iterator)
            except StopIteration:
                logger.warning("后端 %s 返回了空流", provider)
                return provider, iter(())
            except Exception as exc:  # noqa: BLE001 - 逐个降级
                last = exc
                errors.append(f"{provider}: {summarize_exception(exc)}")
                self._degrade(provider, chain[index + 1] if index + 1 < len(chain) else "", exc)
                continue

            def generator(head: str = first, rest: Iterator[str] = iterator) -> Iterator[str]:
                yield head
                yield from rest

            return provider, generator()
        # 全挂：与 chat() 一致，**不**给假答案（t120 F1）
        raise LLMError("所有大模型后端都不可用 -> " + " | ".join(errors),
                       reason=failure_reason(last) if last is not None else "unknown")

    # ---------- 运维 ----------
    def health(self) -> list[dict]:
        report = []
        for provider in self.chain():
            try:
                report.append(self.client_for(provider).health())
            except Exception as exc:  # noqa: BLE001
                report.append({"provider": provider, "error": str(exc)})
        return report
