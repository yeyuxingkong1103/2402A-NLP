"""本地 OpenAI 兼容语音识别服务（CPU，无 vLLM 时的替代方案）。

## 为什么需要它

`研发/scripts/run_whisper.sh` 是**算力云上的正式方案**（vLLM 起 Whisper）；
本机没装 vLLM（Whisper 的 vLLM 部署同样需要它），导致 RAG 的语音链路只能靠内置的 local 后端，
而 local 后端**只支持 WAV**（无 ffmpeg 无法解码 mp3/m4a）。

本脚本提供一个 HTTP 服务，RAG 侧走 **api 后端**调用它，
因此 mp3/m4a 等格式也能被识别（服务端用 transformers 的音频解码，
本机缺 ffmpeg 时同样只支持 WAV —— 这一点如实说明，不夸大）。

    GET  /v1/models                   模型列表（RAG 侧可用性探测）
    POST /v1/audio/transcriptions     语音转写（multipart/form-data）

用法：
    python 研发/scripts/run_local_asr.py
    python 研发/scripts/run_local_asr.py --model models/whisper-tiny --port 8001
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "研发"
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _candidate in (SOURCE_ROOT, PROJECT_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from app.core.logging_conf import logger, setup_logging  # noqa: E402

try:
    from aiohttp import web
except ImportError:  # pragma: no cover
    print("[错误] 缺少 aiohttp：pip install aiohttp", file=sys.stderr)
    raise SystemExit(2)


class LocalWhisper:
    """把 transformers Whisper 包装成 OpenAI 风格的转写器。"""

    def __init__(self, model_path: str, model_name: str) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.pipeline = None
        self.load_seconds = 0.0

    def load(self) -> None:
        from transformers import pipeline

        started = time.perf_counter()
        logger.info("scripts.run_local_asr", "开始加载 Whisper", path=self.model_path)
        self.pipeline = pipeline(
            "automatic-speech-recognition", model=self.model_path, device=-1
        )
        self.load_seconds = time.perf_counter() - started
        logger.info(
            "scripts.run_local_asr",
            "Whisper 加载完成",
            elapsed_s=round(self.load_seconds, 2),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def decode_wav(payload: bytes):
        """把 PCM WAV 解成 float32 数组（绕开 ffmpeg，本机必需）。"""
        import numpy as np

        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())

        if width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif width == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"不支持的 WAV 位宽: {width * 8} bit")
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, rate

    def transcribe(self, payload: bytes, language: str | None = None) -> dict:
        array, rate = self.decode_wav(payload)
        kwargs: dict[str, object] = {}
        if language:
            kwargs["generate_kwargs"] = {"language": language}
        started = time.perf_counter()
        output = self.pipeline({"array": array, "sampling_rate": rate}, **kwargs)
        text = output.get("text", "") if isinstance(output, dict) else str(output)
        return {
            "text": str(text).strip(),
            "language": language or "",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


class Service:
    """HTTP 层：OpenAI 转写接口的最小实现。"""

    def __init__(self, whisper: LocalWhisper) -> None:
        self.whisper = whisper

    async def models(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.whisper.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local-whisper",
                    }
                ],
            }
        )

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "model": self.whisper.model_name})

    async def transcribe(self, request: web.Request) -> web.Response:
        """POST /v1/audio/transcriptions —— multipart 上传，返回 JSON 文本。"""
        try:
            reader = await request.multipart()
        except Exception as exc:
            return web.json_response({"error": {"message": f"请求不是 multipart：{exc}"}}, status=400)

        payload: bytes | None = None
        language = (request.query.get("language") or "").strip() or None
        filename = "audio.wav"
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "file":
                filename = part.filename or filename
                payload = await part.read()
            elif part.name == "language":
                language = (await part.text()).strip() or None
            elif part.name == "response_format":
                await part.read()
            else:
                await part.read()

        if not payload:
            return web.json_response({"error": {"message": "缺少 file 字段"}}, status=400)

        try:
            result = self.whisper.transcribe(payload, language)
        except Exception as exc:
            logger.exception("scripts.run_local_asr", "转写失败", file=filename)
            return web.json_response(
                {"error": {"message": f"{type(exc).__name__}: {exc}"}}, status=422
            )

        logger.info(
            "scripts.run_local_asr",
            "转写完成",
            file=filename,
            bytes=len(payload),
            chars=len(result["text"]),
            elapsed_ms=result["elapsed_ms"],
        )
        # 同时给出 text 与 verbose_json 常用字段，兼容两种 response_format
        return web.json_response(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 OpenAI 兼容语音识别服务（GPU/CPU）")
    parser.add_argument("--model", default="models/whisper-tiny", help="本地 Whisper 目录")
    parser.add_argument("--name", default=None, help="对外模型名（默认取目录名）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--force", action="store_true", help="端口被占用时自动结束占用进程")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    # 端口检查放在加载模型之前：否则端口被占用时要先白等十几秒加载 Whisper，
    # 最后才在 bind 处抛 OSError 10048。
    from port_utils import ensure_port_available

    early = ensure_port_available(
        args.host,
        args.port,
        service_name="语音识别服务",
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

    whisper = LocalWhisper(model_path, model_name)
    try:
        whisper.load()
    except Exception as exc:
        print(f"[错误] Whisper 加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    service = Service(whisper)
    app = web.Application()
    app.router.add_get("/v1/models", service.models)
    app.router.add_post("/v1/audio/transcriptions", service.transcribe)
    app.router.add_get("/health", service.health)

    print("=" * 66)
    print(" 本地 OpenAI 兼容语音识别服务（GPU/CPU 联调，生产请用 run_whisper.sh）")
    print("=" * 66)
    print(f"   模型      : {model_name}")
    print(f"   模型路径  : {model_path}")
    print(f"   加载耗时  : {whisper.load_seconds:.1f}s")
    print(f"   监听      : http://{args.host}:{args.port}")
    print()
    print("   RAG 侧配置（无需改代码）：")
    print(f"     $env:RAG_ASR__BACKEND  = 'api'")
    print(f"     $env:RAG_ASR__BASE_URL = 'http://{args.host}:{args.port}/v1'")
    print(f"     $env:RAG_ASR__MODEL    = '{model_name}'")
    print()
    print("   自检： curl http://127.0.0.1:%d/v1/models" % args.port)
    print("   注意：本机无 ffmpeg，只支持 WAV；mp3/m4a 需在算力云上运行。")
    print("=" * 66, flush=True)

    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
