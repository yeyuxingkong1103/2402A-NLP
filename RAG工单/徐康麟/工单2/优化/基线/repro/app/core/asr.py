"""语音识别（ASR）：把用户录制的音频转写成文字。

工单追加要求：**支持语音输入**。

三种转写后端，按以下顺序自动选择（可用 ``RAG_ASR__BACKEND`` 强制指定）:

1. ``api``    —— 调用 OpenAI 兼容的 ``/v1/audio/transcriptions`` 接口。
                 算力云上用 vLLM 起一个 Whisper 服务即可（见 ``scripts/run_whisper.sh``），
                 与 LLM 共用同一套 OpenAI 协议，无需额外客户端。
2. ``local``  —— 本地 transformers Whisper（``models/whisper-*``）。
                 完全离线可用，但 4090 上仍建议走 API 以省显存。
3. ``none``   —— 均不可用时明确报错，**绝不静默返回空字符串**
                 （否则用户会以为“说了话但系统没反应”）。

设计约束：
- 中文与英文都要支持，``language`` 传 ``None`` 时由模型自动判别；
- 所有失败都带原因，供界面中文提示；
- ``@trace`` 记录输入输出，符合工单日志要求。
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging_conf import logger, trace

try:  # pragma: no cover - 依赖可用性分支
    from openai import OpenAI

    HAS_OPENAI = True
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    HAS_OPENAI = False


class TranscriptionError(RuntimeError):
    """语音转写失败（带用户可读的中文原因）。"""


class SpeechRecognizer:
    """语音转写器。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._local_pipeline = None
        self._local_model_name = ""
        self._client = None
        self.last_error = ""

    # ------------------------------------------------------------------
    # 后端探测
    # ------------------------------------------------------------------
    def local_model_dir(self) -> Path | None:
        """查找本地 Whisper 模型目录（models/whisper-*）。"""
        configured = self.settings.asr.local_model_dir
        if configured:
            path = PROJECT_ROOT / configured
            if path.is_dir():
                return path
        models_root = PROJECT_ROOT / "models"
        if models_root.is_dir():
            for candidate in sorted(models_root.glob("whisper*")):
                if candidate.is_dir():
                    return candidate
        return None

    def available_backends(self) -> dict[str, bool]:
        """返回各后端的可用性，供界面展示与自检。"""
        return {
            "api": HAS_OPENAI and bool(self.settings.asr.base_url),
            "local": self.local_model_dir() is not None,
        }

    def resolve_backend(self) -> str:
        """决定实际使用的后端名（api / local / none）。"""
        configured = (self.settings.asr.backend or "auto").lower()
        available = self.available_backends()
        if configured in {"api", "local"}:
            return configured if available.get(configured) else "none"
        # auto：优先 API（省显存），其次本地模型
        if available["api"]:
            return "api"
        if available["local"]:
            return "local"
        return "none"

    # ------------------------------------------------------------------
    # 转写
    # ------------------------------------------------------------------
    @trace
    def transcribe(self, audio: bytes | Path | str, language: str | None = None) -> dict[str, object]:
        """把音频转写成文字。

        Args:
            audio: 音频字节流，或音频文件路径。
            language: ``"zh"`` / ``"en"``；``None`` 表示自动判别（中英混说时推荐）。

        Returns:
            ``{"text": 文本, "language": 语言, "backend": 后端, "elapsed_ms": 耗时}``

        Raises:
            TranscriptionError: 无可用的转写后端，或转写失败。
        """
        started = time.perf_counter()
        backend = self.resolve_backend()
        if backend == "none":
            detail = "未配置语音服务（OpenAI 兼容接口不可用且没有本地 Whisper 模型）"
            self.last_error = detail
            raise TranscriptionError(
                f"{detail}。请在算力云上启动 Whisper 服务（scripts/run_whisper.sh），"
                f"或把 Whisper 模型放到 models/ 目录下。"
            )

        if backend == "api":
            text, detected = self._transcribe_api(audio, language)
        else:
            text, detected = self._transcribe_local(audio, language)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        result = {
            "text": (text or "").strip(),
            "language": detected or language or "",
            "backend": backend,
            "elapsed_ms": elapsed_ms,
        }
        if not result["text"]:
            # 空转写必须有明确原因，不能让上层拿到“空问题”再去检索
            self.last_error = "音频未识别出任何文字（可能是静音、噪声过大或语速过快）"
            logger.warning("app.core.asr", "语音转写结果为空", backend=backend, elapsed_ms=elapsed_ms)
            raise TranscriptionError(self.last_error)
        logger.info(
            "app.core.asr",
            "语音转写完成",
            backend=backend,
            language=result["language"],
            chars=len(str(result["text"])),
            elapsed_ms=elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    def _as_file_tuple(self, audio: bytes | Path | str) -> tuple[str, bytes]:
        """把音频统一成 ``(文件名, 字节)``，供 OpenAI SDK 上传。"""
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if not path.exists():
                raise TranscriptionError(f"音频文件不存在: {path}")
            return path.name, path.read_bytes()
        return "audio.wav", bytes(audio)

    def _transcribe_api(self, audio: bytes | Path | str, language: str | None) -> tuple[str, str]:
        """调用 OpenAI 兼容的转写接口。"""
        if not HAS_OPENAI:
            raise TranscriptionError("未安装 openai 库，无法调用语音接口")
        if self._client is None:
            self._client = OpenAI(
                base_url=self.settings.asr.base_url,
                api_key=self.settings.asr.api_key,
                timeout=self.settings.asr.timeout,
            )
        filename, payload = self._as_file_tuple(audio)
        try:
            buffer = io.BytesIO(payload)
            buffer.name = filename  # SDK 依赖 name 推断 MIME
            kwargs: dict[str, object] = {
                "model": self.settings.asr.model,
                "file": buffer,
                "response_format": "verbose_json",
            }
            if language:
                kwargs["language"] = language
            try:
                response = self._client.audio.transcriptions.create(**kwargs)  # type: ignore[arg-type]
            except Exception:
                # 部分服务不支持 verbose_json，退回纯文本
                kwargs["response_format"] = "json"
                response = self._client.audio.transcriptions.create(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("app.core.asr", "调用语音接口失败", base_url=self.settings.asr.base_url)
            raise TranscriptionError(f"调用语音识别服务失败：{self.last_error}") from exc

        text = getattr(response, "text", None)
        if text is None and isinstance(response, dict):
            text = response.get("text", "")
        detected = getattr(response, "language", "") or ""
        return str(text or ""), str(detected)

    def _transcribe_local(self, audio: bytes | Path | str, language: str | None) -> tuple[str, str]:
        """用本地 transformers Whisper 转写。

        关键点：**不把文件路径交给 pipeline**。
        transformers 的音频后端依赖 ffmpeg；本机（以及很多 Windows 环境）
        没有 ffmpeg，直接传路径会报
        ``ValueError: ffmpeg was not found but is required to load audio files``。
        因此这里先用标准库 ``wave`` 把 PCM WAV 解成 numpy 数组，
        再以 ``{"array": ..., "sampling_rate": ...}`` 形式喂给 pipeline——
        完全绕开 ffmpeg，离线可用。
        非 WAV（mp3/m4a）仍需 ffmpeg，此时给出明确提示并建议改用 API 后端。
        """
        model_dir = self.local_model_dir()
        if model_dir is None:
            raise TranscriptionError("未找到本地 Whisper 模型（models/whisper-*）")

        # ---- 1. 解码音频（优先标准库 wave，绕开 ffmpeg）----
        array, sample_rate = self._decode_audio(audio)

        try:
            if self._local_pipeline is None or self._local_model_name != str(model_dir):
                from transformers import pipeline  # 延迟导入，避免拖慢启动

                started = time.perf_counter()
                self._local_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=str(model_dir),
                    device=-1,  # CPU；算力云上建议改用 API 后端
                )
                self._local_model_name = str(model_dir)
                logger.info(
                    "app.core.asr",
                    "本地 Whisper 模型加载完成",
                    model=str(model_dir),
                    elapsed_s=round(time.perf_counter() - started, 2),
                )
            kwargs: dict[str, object] = {}
            if language:
                kwargs["generate_kwargs"] = {"language": language}
            output = self._local_pipeline(
                {"array": array, "sampling_rate": sample_rate}, **kwargs  # type: ignore[operator]
            )
            text = output.get("text", "") if isinstance(output, dict) else str(output)
            return str(text), language or ""
        except TranscriptionError:
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("app.core.asr", "本地语音转写失败", model=str(model_dir))
            raise TranscriptionError(f"本地语音识别失败：{self.last_error}") from exc

    def _decode_audio(self, audio: bytes | Path | str) -> tuple["object", int]:
        """把音频解码成 ``(numpy float32 数组, 采样率)``。

        只支持 PCM WAV（标准库 ``wave``）——浏览器的 ``st.audio_input``
        与常见录音工具默认就是 WAV，足够覆盖主要场景。
        """
        import numpy as np

        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if not path.exists():
                raise TranscriptionError(f"音频文件不存在: {path}")
            payload = path.read_bytes()
        else:
            payload = bytes(audio)

        try:
            import io
            import wave

            with wave.open(io.BytesIO(payload), "rb") as handle:
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
        except Exception as exc:
            raise TranscriptionError(
                "只支持 WAV 格式的本地识别（未检测到 ffmpeg，无法解码 mp3/m4a）。"
                "请改用 WAV，或在算力云上启动 Whisper 服务后使用 API 后端。"
                f"（原始错误：{type(exc).__name__}）"
            ) from exc

        if width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif width == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise TranscriptionError(f"不支持的 WAV 采样位宽: {width * 8} bit")

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, rate

    # ------------------------------------------------------------------
    def health(self) -> dict[str, object]:
        """语音链路健康状态。"""
        backends = self.available_backends()
        resolved = self.resolve_backend()
        model_dir = self.local_model_dir()
        return {
            "backend": resolved,
            "configured": self.settings.asr.backend,
            "api_available": backends["api"],
            "local_available": backends["local"],
            "local_model": str(model_dir) if model_dir else "",
            "base_url": self.settings.asr.base_url,
            "model": self.settings.asr.model,
            "last_error": self.last_error,
        }


_recognizer: SpeechRecognizer | None = None


def get_speech_recognizer() -> SpeechRecognizer:
    """工厂函数：获取进程级单例。"""
    global _recognizer
    if _recognizer is None:
        _recognizer = SpeechRecognizer()
    return _recognizer


def reset_speech_recognizer() -> None:
    """重置单例（测试用）。"""
    global _recognizer
    _recognizer = None
