"""Qwen3 本地生成：chat 模板 + 流式输出 + 单卡串行保护。

* 模型路径默认取自 ``paths.models_root/Qwen3-0.6B``；
* Qwen3 的思考模式通过 ``enable_thinking`` 控制，0.6B 默认关闭以保证稳定；
* 同一时刻只允许一个生成任务占用 GPU（``threading.Lock``），
  多用户并发时按到达顺序排队，避免显存抖动。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import torch

from ..config import Config, get_config
from ..errors import DependencyError, ValidationError
from ..logging_conf import get_logger

logger = get_logger(__name__)

_DTYPES = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32, "fp32": torch.float32}


@dataclass
class GenerationResult:
    """一次生成的统计信息。"""

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    finish_reason: str = "stop"
    param: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed": round(self.elapsed, 3),
            "finish_reason": self.finish_reason,
            "param": self.param,
        }


class LocalLLM:
    """本地因果语言模型封装。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        section = self.config.section("models.llm")
        self.model_path = self.config.llm_path()
        self.max_input_tokens = int(section.get("max_input_tokens", 7000))
        self.defaults = {
            "max_new_tokens": int(section.get("max_new_tokens", 700)),
            "temperature": float(section.get("temperature", 0.35)),
            "top_p": float(section.get("top_p", 0.85)),
            "repetition_penalty": float(section.get("repetition_penalty", 1.08)),
        }
        self.enable_thinking = bool(section.get("enable_thinking", False))
        self._lock = threading.Lock()          # 生成串行锁（保护 GPU）
        self._load_lock = threading.RLock()     # 加载串行锁（防止并发首载）
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    # ------------------------------------------------------------------ 加载
    def load(self) -> "LocalLLM":
        # 与嵌入模型同理：加载过程必须串行，避免并发首载导致半初始化
        with self._load_lock:
            if self._loaded:
                return self
            if not self.model_path.is_dir():
                raise DependencyError(f"语言模型目录不存在：{self.model_path}")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            section = self.config.section("models.llm")
            device = str(section.get("device", "cuda"))
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，LLM 回退到 CPU（会很慢）")
                device = "cpu"
            dtype = _DTYPES.get(str(section.get("dtype", "float16")).lower(), torch.float16)
            if device == "cpu":
                dtype = torch.float32

            logger.info("加载语言模型：%s (device=%s, dtype=%s)", self.model_path, device, dtype)
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            try:
                model = AutoModelForCausalLM.from_pretrained(str(self.model_path), dtype=dtype)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(str(self.model_path), torch_dtype=dtype)
            model.to(device)
            model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self.device = device
            self._loaded = True
            logger.info("语言模型就绪：%s", self.model_path.name)
            return self

    # ------------------------------------------------------------------ 模板
    def build_prompt(self, messages: Sequence[dict]) -> str:
        """按 chat 模板渲染提示词（便于日志与调试面板查看）。"""

        self.load()
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        try:
            return self.tokenizer.apply_chat_template(
                list(messages), enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(list(messages), **kwargs)

    def _encode(self, messages: Sequence[dict]) -> torch.Tensor:
        kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        try:
            encoded = self.tokenizer.apply_chat_template(
                list(messages), enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        # transformers 5.x 返回 BatchEncoding，旧版本直接返回 Tensor
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if input_ids.shape[1] > self.max_input_tokens:
            head = input_ids[:, :1]
            tail = input_ids[:, -(self.max_input_tokens - 1):]
            logger.warning("提示词超长（%d tokens），已截断到 %d", input_ids.shape[1], self.max_input_tokens)
            input_ids = torch.cat([head, tail], dim=1)
        return input_ids.to(self.device)

    def _gen_kwargs(self, overrides: dict | None) -> dict:
        params = dict(self.defaults)
        for key, value in (overrides or {}).items():
            if value is not None and key in params:
                params[key] = value
        temperature = max(float(params["temperature"]), 0.0)
        kwargs = {
            "max_new_tokens": int(params["max_new_tokens"]),
            "repetition_penalty": float(params["repetition_penalty"]),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(params["top_p"])
        return kwargs

    # ------------------------------------------------------------------ 生成
    def complete(self, messages: Sequence[dict], **overrides) -> GenerationResult:
        """非流式生成（用于摘要、查询改写等内部任务）。"""

        if not messages:
            raise ValidationError("messages 不能为空")
        self.load()
        params = self._gen_kwargs(overrides)
        started = time.time()
        with self._lock:
            input_ids = self._encode(messages)
            with torch.no_grad():
                output = self.model.generate(input_ids=input_ids, **params)
        new_tokens = output[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=int(input_ids.shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
            elapsed=time.time() - started,
            param={k: v for k, v in params.items() if k != "pad_token_id"},
        )

    def stream(
        self,
        messages: Sequence[dict],
        on_done: Callable[[GenerationResult], None] | None = None,
        **overrides,
    ) -> Iterator[str]:
        """流式生成：逐段产出文本；结束后回调 on_done 返回统计信息。"""

        if not messages:
            raise ValidationError("messages 不能为空")
        self.load()
        from transformers import TextIteratorStreamer

        params = self._gen_kwargs(overrides)
        text_parts: list[str] = []
        started = time.time()
        self._lock.acquire()
        try:
            input_ids = self._encode(messages)
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600.0
            )
            error: list[BaseException] = []

            def _run() -> None:
                try:
                    with torch.no_grad():
                        self.model.generate(input_ids=input_ids, streamer=streamer, **params)
                except BaseException as exc:  # pragma: no cover - 线程内异常透传
                    error.append(exc)
                    streamer.end()

            thread = threading.Thread(target=_run, name="role-rag-llm", daemon=True)
            thread.start()
            for chunk in streamer:
                if chunk:
                    text_parts.append(chunk)
                    yield chunk
            thread.join(timeout=5)
            if error:
                raise error[0]

            full = "".join(text_parts)
            result = GenerationResult(
                text=full,
                prompt_tokens=int(input_ids.shape[1]),
                completion_tokens=len(self.tokenizer.encode(full, add_special_tokens=False)) if full else 0,
                elapsed=time.time() - started,
                finish_reason="stop",
                param={k: v for k, v in params.items() if k != "pad_token_id"},
            )
            if on_done is not None:
                on_done(result)
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ 工具
    def count_tokens(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode(text or "", add_special_tokens=False))

    def stats(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "loaded": self._loaded,
            "device": self.device,
            "enable_thinking": self.enable_thinking,
            "defaults": dict(self.defaults),
        }


_llm: LocalLLM | None = None
_llm_lock = threading.Lock()


def get_llm(config: Config | None = None) -> LocalLLM:
    """获取（并懒加载）全局语言模型单例。"""

    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = LocalLLM(config)
        return _llm
