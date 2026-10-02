"""本地 OpenAI 兼容推理服务（CPU，无 vLLM 时的替代方案）。

## 为什么需要它

`scripts/run_vllm.sh` 是**算力云 4090 上的正式方案**；本机虽有 GPU
（RTX 2060，已装 CUDA 版 torch），但没有 vLLM（其 Windows 支持需 WSL），导致：

- RAG 系统探测不到 LLM 服务，只能走“抽取式回答”降级路径；
- **LLM 流式生成那条链路（首字延迟、SSE 增量、引用回填）在本地无法验证**。

本脚本用纯 Python（aiohttp + transformers）实现一个最小但**协议兼容**的
推理服务，让完整链路可以在本地跑通：

    GET  /v1/models              模型列表（RAG 侧用它做可用性探测）
    POST /v1/chat/completions    对话补全，支持 stream=true/false

协议对齐 OpenAI，因此 RAG 侧**不需要改任何代码**，只要把
``RAG_LLM__BASE_URL`` 指过来即可。

## 与 vLLM 的差距（如实说明）

| 维度 | 本服务 | vLLM（4090） |
| --- | --- | --- |
| 设备 | CPU | GPU |
| 模型 | 本地小模型（默认 Qwen3-0.6B） | Qwen2.5-7B-Instruct-AWQ |
| 并发 | 串行（有锁，避免 CPU 争抢） | 连续批处理，中等并发 |
| 首字延迟 | 0.3~2 秒（模型越小越快） | 通常 < 300ms |

**它只用于本地链路联调，不作为生产服务。**

用法：
    python 研发/scripts/run_local_llm.py
    python 研发/scripts/run_local_llm.py --model models/Qwen3-0.6B --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "研发"
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _candidate in (SOURCE_ROOT, PROJECT_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from app.core.config import get_settings  # noqa: E402
from app.core.logging_conf import logger, setup_logging  # noqa: E402

try:
    from aiohttp import web
except ImportError:  # pragma: no cover
    print("[错误] 缺少 aiohttp，无法启动本地推理服务：pip install aiohttp", file=sys.stderr)
    raise SystemExit(2)


class LocalLLM:
    """把 transformers 因果语言模型包装成 OpenAI 风格的补全器。"""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        max_new_tokens: int = 512,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.requested_device = device
        self.requested_dtype = dtype
        self.tokenizer = None
        self.model = None
        self.load_seconds = 0.0
        self.device = "cpu"
        # CPU 上并发推理会互相抢核反而更慢，用锁串行化；
        # GPU 上允许并发（KV cache 由 PyTorch 自行管理），故不加锁。
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        """决定推理设备：auto 时优先 CUDA。"""
        import torch

        if self.requested_device != "auto":
            return self.requested_device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self):
        """决定权重精度。

        GPU 上用 float16（RTX 20 系不支持 bfloat16，6GB 显存也放不下 float32）；
        CPU 上保持 float32（CPU 对 float16 支持很差）。
        """
        import torch

        if self.requested_dtype == "float16":
            return torch.float16
        if self.requested_dtype == "bfloat16":
            return torch.bfloat16
        if self.requested_dtype == "float32":
            return torch.float32
        return torch.float16 if self.device.startswith("cuda") else torch.float32

    def load(self) -> None:
        """加载模型（阻塞，启动时执行一次）。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.device = self._resolve_device()
        dtype = self._resolve_dtype()
        logger.info(
            "scripts.run_local_llm",
            "开始加载模型",
            path=self.model_path,
            device=self.device,
            dtype=str(dtype).replace("torch.", ""),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

        vram = ""
        if self.device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram = f"{allocated:.2f}/{total:.2f} GB"
        logger.info(
            "scripts.run_local_llm",
            "模型加载完成",
            model=self.model_name,
            device=self.device,
            vram=vram,
            elapsed_s=round(self.load_seconds, 2),
            params_m=round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1),
        )

    # ------------------------------------------------------------------
    def _render(self, messages: list[dict[str, str]]) -> str:
        """套用对话模板；Qwen3 需显式关闭思考模式，否则 token 会被 <think> 吃光。"""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _inputs(self, messages: list[dict[str, str]]):
        """构造模型输入，并把张量放到目标设备（GPU 时必须）。"""
        rendered = self._render(messages)
        encoded = self.tokenizer(rendered, return_tensors="pt")
        prompt_len = int(encoded["input_ids"].shape[1])
        if self.device.startswith("cuda"):
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
        return encoded, prompt_len

    @staticmethod
    def _strip_think(text: str) -> str:
        import re

        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _eos_token_id(self) -> int | None:
        """取 EOS id：生成时必须让模型有机会停下来，否则会一直吐到 max_new_tokens。"""
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            return int(eos)
        generation = getattr(self.model, "generation_config", None)
        return int(generation.eos_token_id) if generation and generation.eos_token_id else None

    def _decode(self, output, prompt_len: int) -> str:
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        on_delta=None,
    ) -> tuple[str, float]:
        """生成补全。

        Args:
            on_delta: 传入回调时走**逐 token 流式**（用于 SSE）；
                      否则一次性生成（更快，但无法测首字延迟）。

        Returns:
            ``(完整文本, 首字耗时毫秒)``
        """
        import torch
        from transformers import TextIteratorStreamer

        # CPU 串行执行（并发只会互相抢核）；GPU 放开并发
        lock = self._lock if self.device == "cpu" else _NullLock()
        async with lock:
            inputs, prompt_len = self._inputs(messages)
            limit = max(1, min(max_tokens or self.max_new_tokens, self.max_new_tokens))
            generate_kwargs = {
                "max_new_tokens": limit,
                "do_sample": temperature > 0,
                "temperature": max(temperature, 1e-5),
                "repetition_penalty": 1.05,
            }
            eos = self._eos_token_id()
            if eos is not None:
                generate_kwargs["eos_token_id"] = eos
                generate_kwargs["pad_token_id"] = eos

            if on_delta is None:
                started = time.perf_counter()
                with torch.no_grad():
                    output = self.model.generate(**inputs, **generate_kwargs)
                text = self._decode(output, prompt_len)
                return self._strip_think(text), (time.perf_counter() - started) * 1000

            # 流式：TextIteratorStreamer + 后台线程
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs["streamer"] = streamer
            loop = asyncio.get_running_loop()

            def _run() -> None:
                with torch.no_grad():
                    self.model.generate(**inputs, **generate_kwargs)

            started = time.perf_counter()
            first_token_ms = 0.0
            pieces: list[str] = []
            task = loop.run_in_executor(None, _run)
            iterator = iter(streamer)
            while True:
                piece = await loop.run_in_executor(None, next, iterator, None)
                if piece is None:
                    break
                if not piece:
                    continue
                if first_token_ms == 0.0:
                    first_token_ms = (time.perf_counter() - started) * 1000
                pieces.append(piece)
                await on_delta(piece)
            await task
            return self._strip_think("".join(pieces)), (first_token_ms or (time.perf_counter() - started) * 1000)


class _NullLock:
    """GPU 上使用的空锁（异步上下文管理器，不做任何同步）。"""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class Service:
    """HTTP 层：OpenAI 兼容的最小实现。"""

    def __init__(self, llm: LocalLLM) -> None:
        self.llm = llm
        self.stats = {"requests": 0, "tokens": 0, "last_first_token_ms": 0.0}

    async def models(self, _request: web.Request) -> web.Response:
        """GET /v1/models —— RAG 侧用它判断服务是否可用。"""
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.llm.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local-transformers",
                    }
                ],
            }
        )

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "model": self.llm.model_name, **self.stats})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        """POST /v1/chat/completions —— 支持 stream=true/false。"""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "请求体不是合法 JSON"}}, status=400)

        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return web.json_response(
                {"error": {"message": "messages 不能为空"}}, status=400
            )
        stream = bool(body.get("stream"))
        max_tokens = int(body.get("max_tokens") or self.llm.max_new_tokens)
        temperature = float(body.get("temperature") or 0.0)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        self.stats["requests"] += 1

        if not stream:
            text, elapsed_ms = await self.llm.complete(messages, max_tokens, temperature)
            self.stats["last_first_token_ms"] = round(elapsed_ms, 1)
            return web.json_response(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": self.llm.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )

        # ---- 流式（SSE）----
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        def frame(delta: dict, finish: str | None = None) -> bytes:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.llm.model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        async def on_delta(piece: str) -> None:
            self.stats["tokens"] += 1
            await response.write(frame({"content": piece}))

        try:
            text, first_token_ms = await self.llm.complete(
                messages, max_tokens, temperature, on_delta=on_delta
            )
            self.stats["last_first_token_ms"] = round(first_token_ms, 1)
            await response.write(frame({}, "stop"))
            await response.write(b"data: [DONE]\n\n")
        except Exception as exc:  # pragma: no cover - 生成异常
            logger.exception("scripts.run_local_llm", "流式生成失败", error=str(exc))
            err = {"error": {"message": f"{type(exc).__name__}: {exc}"}}
            await response.write(f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8"))
        await response.write_eof()
        logger.info(
            "scripts.run_local_llm",
            "流式完成",
            chars=len(text),
            first_token_ms=round(first_token_ms, 1),
        )
        return response


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="本地 OpenAI 兼容推理服务（CPU）")
    parser.add_argument(
        "--model",
        default="models/Qwen3-0.6B",
        help="本地模型目录（相对项目根）或 HuggingFace 模型名",
    )
    parser.add_argument("--name", default=None, help="对外暴露的模型名（默认取目录名）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda:0")
    parser.add_argument("--dtype", default="auto", help="auto / float16 / float32")
    parser.add_argument("--force", action="store_true", help="端口被占用时自动结束占用进程")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    # 端口检查必须在**加载模型之前**：
    # 否则端口被占用时要先白等十几秒加载模型，最后才在 bind 处抛
    # OSError: [Errno 10048]，体验很差且浪费显存。
    from port_utils import ensure_port_available

    early = ensure_port_available(
        args.host,
        args.port,
        service_name="LLM 推理服务",
        probe_path="/v1/models",
        force=args.force,
    )
    if early is not None:
        return early

    model_path = args.model
    candidate = PROJECT_ROOT / args.model
    if candidate.exists():
        model_path = str(candidate)
    model_name = args.name or Path(model_path).name

    llm = LocalLLM(
        model_path,
        model_name,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    try:
        llm.load()
    except Exception as exc:
        print(f"[错误] 模型加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    service = Service(llm)
    app = web.Application()
    app.router.add_get("/v1/models", service.models)
    app.router.add_post("/v1/chat/completions", service.chat)
    app.router.add_get("/health", service.health)

    print("=" * 66)
    print(" 本地 OpenAI 兼容推理服务（GPU/CPU 联调用，生产请用 vLLM）")
    print("=" * 66)
    print(f"   模型      : {model_name}")
    print(f"   模型路径  : {model_path}")
    # 明确显示推理设备：这是「是否真的用上 GPU」最直接的判据
    device_desc = llm.device
    if llm.device.startswith("cuda"):
        import torch

        device_desc = (
            f"{llm.device} ({torch.cuda.get_device_name(0)}, "
            f"显存 {torch.cuda.memory_allocated(0) / 1024**3:.2f}/"
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)"
        )
    else:
        device_desc = "cpu（首字延迟会明显偏高，属预期）"
    print(f"   推理设备  : {device_desc}")
    print(f"   加载耗时  : {llm.load_seconds:.1f}s")
    print(f"   监听      : http://{args.host}:{args.port}")
    print()
    print("   RAG 侧配置（无需改代码）：")
    print(f"     $env:RAG_LLM__BASE_URL = 'http://{args.host}:{args.port}/v1'")
    print(f"     $env:RAG_LLM__MODEL    = '{model_name}'")
    print()
    print("   自检： curl http://127.0.0.1:%d/v1/models" % args.port)
    print("=" * 66, flush=True)

    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
