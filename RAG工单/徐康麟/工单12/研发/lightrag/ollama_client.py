# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""Ollama 客户端：只用标准库（urllib + json），避免额外依赖。

提供三个能力：
  * :func:`embed_texts` —— ``/api/embed`` 批量取向量（bge-m3，1024 维）
  * :func:`generate`    —— ``/api/generate`` 非流式生成
  * :func:`generate_stream` —— 流式生成（CLI 打字机效果）
  * :func:`health`      —— 探活，模型缺失时给出明确报错而不是裸崩

所有耗时/token 计数都通过 :class:`CallStats` 回传，供评估脚本统计真实数字。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .config import Cfg


class LLMError(RuntimeError):
    """LLM/嵌入调用失败（已重试过）。"""


@dataclass
class CallStats:
    """单次（或累计）调用的真实统计。"""

    calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    failures: list[str] = field(default_factory=list)

    def add(self, other: "CallStats") -> None:
        self.calls += other.calls
        self.seconds += other.seconds
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.retries += other.retries
        self.failures.extend(other.failures)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "seconds": round(self.seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "retries": self.retries,
            "failures": self.failures[:20],
        }


def _post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{endpoint}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body) if body.strip() else {}


def _get_json(base_url: str, endpoint: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{endpoint}", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body) if body.strip() else {}


class OllamaClient:
    """极简 Ollama 客户端；失败时按配置重试，最终抛 :class:`LLMError`。"""

    def __init__(self, cfg: Cfg, *, logger: Any = None) -> None:
        self.cfg = cfg
        self.llm = cfg.llm
        self.base_url = str(self.llm.base_url)
        self.log = logger
        self.stats = CallStats()

    # ---------------------------------------------------------------- 嵌入
    def embed_texts(self, texts: Sequence[str], *, batch_size: int | None = None,
                    model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        bs = int(batch_size or self.llm.embed_batch_size)
        mdl = model or str(self.llm.embed_model)
        timeout = float(self.llm.embed_timeout_s)
        out: list[list[float]] = []
        for start in range(0, len(texts), bs):
            batch = list(texts[start:start + bs])
            payload = {"model": mdl, "input": batch}
            t0 = time.perf_counter()
            data = self._with_retry(
                lambda p=payload: _post_json(self.base_url, "/api/embed", p, timeout),
                what=f"embed[{start}:{start + len(batch)}]",
            )
            self.stats.seconds += time.perf_counter() - t0
            self.stats.calls += 1
            got = data.get("embeddings") or []
            if len(got) != len(batch):
                raise LLMError(f"Ollama 嵌入返回条数不符：期望 {len(batch)}，实际 {len(got)}")
            out.extend([list(map(float, vec)) for vec in got])
        return out

    def embed_query(self, text: str) -> list[float]:
        vecs = self.embed_texts([text])
        return vecs[0] if vecs else []

    # ---------------------------------------------------------------- 生成
    def generate(self, prompt: str, *, model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, kind: str = "gen") -> str:
        text, _ = self.generate_with_stats(prompt, model=model, temperature=temperature,
                                           max_tokens=max_tokens, kind=kind)
        return text

    def generate_with_stats(self, prompt: str, *, model: str | None = None,
                            temperature: float | None = None, max_tokens: int | None = None,
                            kind: str = "gen") -> tuple[str, CallStats]:
        """非流式生成，返回 (文本, 本次调用统计)。"""
        local = CallStats()
        if kind == "extract":
            mdl = model or str(self.llm.extract_model)
            temp = float(self.llm.extract_temperature if temperature is None else temperature)
            cap = int(max_tokens or self.llm.extract_max_tokens)
        else:
            mdl = model or str(self.llm.gen_model)
            temp = float(self.llm.temperature if temperature is None else temperature)
            cap = int(max_tokens or self.llm.max_tokens)
        payload = {
            "model": mdl,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp, "num_predict": cap},
            "keep_alive": str(self.llm.keep_alive),
        }
        t0 = time.perf_counter()
        data = self._with_retry(
            lambda: _post_json(self.base_url, "/api/generate", payload, float(self.llm.request_timeout_s)),
            what=f"generate[{mdl}]",
        )
        local.seconds += time.perf_counter() - t0
        local.calls += 1
        local.prompt_tokens = int(data.get("prompt_eval_count") or 0)
        local.completion_tokens = int(data.get("eval_count") or 0)
        self.stats.add(local)
        return str(data.get("response") or ""), local

    def generate_stream(self, prompt: str, *, model: str | None = None, temperature: float | None = None,
                        max_tokens: int | None = None) -> Iterator[str]:
        """流式生成（逐段 yield 文本）。"""
        mdl = model or str(self.llm.gen_model)
        temp = float(self.llm.temperature if temperature is None else temperature)
        cap = int(max_tokens or self.llm.max_tokens)
        payload = {
            "model": mdl, "prompt": prompt, "stream": True,
            "options": {"temperature": temp, "num_predict": cap},
            "keep_alive": str(self.llm.keep_alive),
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.llm.request_timeout_s)) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = obj.get("response") or ""
                    if piece:
                        yield str(piece)
                    if obj.get("done"):
                        self.stats.calls += 1
                        self.stats.prompt_tokens += int(obj.get("prompt_eval_count") or 0)
                        self.stats.completion_tokens += int(obj.get("eval_count") or 0)
                        break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.stats.failures.append(f"stream: {type(exc).__name__}: {exc}")
            raise LLMError(f"Ollama 流式生成失败：{type(exc).__name__}: {exc}") from exc

    # ---------------------------------------------------------------- 探活
    def health(self, *, timeout: float = 5.0) -> dict[str, Any]:
        """返回 {'ok':bool,'models':[...],'missing':[...]}；失败不抛异常。"""
        need = [str(self.llm.gen_model), str(self.llm.extract_model), str(self.llm.embed_model)]
        try:
            data = _get_json(self.base_url, "/api/tags", timeout)
        except Exception as exc:  # noqa: BLE001 - 探活不应崩
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "models": [], "missing": need}
        names = [str(m.get("name", "")) for m in (data.get("models") or [])]
        missing = []
        for want in need:
            base = want.split(":")[0]
            if not any(n == want or n.split(":")[0] == base for n in names):
                missing.append(want)
        return {"ok": not missing, "models": names, "missing": missing, "base_url": self.base_url}

    # ---------------------------------------------------------------- 内部
    def _with_retry(self, fn: Any, *, what: str) -> dict[str, Any]:
        attempts = int(self.llm.max_retries) + 1
        sleep_s = float(self.llm.retry_sleep_s)
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return fn()
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, LLMError) as exc:
                last = exc
                self.stats.retries += 1
                self.stats.failures.append(f"{what} 第{attempt + 1}次: {type(exc).__name__}: {exc}")
                if attempt + 1 < attempts:
                    time.sleep(sleep_s * (attempt + 1))
        raise LLMError(f"{what} 重试 {attempts} 次仍失败：{type(last).__name__}: {last}")
