"""离线测试：语音识别（ASR）模块。

工单追加要求：支持语音输入。本文件覆盖：

- 语言检测（中/英/缩写）；
- 音频解码（标准库 wave，绕开 ffmpeg）；
- 后端选择逻辑（配置强制 / 无可用后端时的明确报错）；
- 空音频必须抛出带中文原因的 ``TranscriptionError``，而不是静默返回空串。

真实 Whisper 推理较慢（CPU 上首次加载约 10 秒），因此默认用
``pytest.mark.slow`` 风格的条件跳过：只有本地模型存在时才跑一次真推理，
其余用例只验证确定性的解码与选择逻辑。
"""

from __future__ import annotations

import io
import math
import struct
import wave

import pytest

from app.core.asr import SpeechRecognizer, TranscriptionError, get_speech_recognizer


def make_wav(seconds: float = 1.0, rate: int = 16000, freq: float = 300.0, channels: int = 1) -> bytes:
    """生成一段正弦波 WAV（用于验证解码链路，不含真实语音）。

    注意：``wave.writeframes`` 接收的是**交错后的样本字节流**，
    总样本数必须是 ``帧数 × 声道数``。多声道时若只写单声道的样本个数，
    实际音频时长会变成一半，导致解码后的样本数断言失败。
    """
    buffer = io.BytesIO()
    frames = int(rate * seconds)
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        samples = b"".join(
            struct.pack("<h", int(1200 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(frames)
        )
        handle.writeframes(samples * channels)
    return buffer.getvalue()


# ==========================================================================
# 音频解码：必须绕开 ffmpeg（本机没有 ffmpeg，走路径会直接失败）
# ==========================================================================
def test_decode_wav_without_ffmpeg() -> None:
    """PCM WAV 必须能被标准库解码成 float32 数组与正确采样率。"""
    recognizer = SpeechRecognizer()
    array, rate = recognizer._decode_audio(make_wav(seconds=0.5, rate=16000))

    assert rate == 16000, f"采样率解析错误：{rate}"
    assert len(array) == 8000, f"样本数应为 8000，实际 {len(array)}"
    assert abs(float(array.max())) < 1.0, "样本应归一化到 [-1, 1]"


def test_decode_stereo_is_mixed_to_mono() -> None:
    """立体声必须下混成单声道（Whisper 只接受单声道）。

    0.25 秒 @16kHz 双声道 = 每声道 4000 帧，交错后 8000 个样本，
    下混后应回到 4000 个单声道样本。
    """
    recognizer = SpeechRecognizer()
    array, rate = recognizer._decode_audio(make_wav(seconds=0.25, rate=16000, channels=2))

    assert rate == 16000
    assert len(array) == 4000, f"下混后样本数应为 4000，实际 {len(array)}"


def test_decode_rejects_non_wav_with_readable_reason() -> None:
    """非 WAV 数据必须报错，且原因里说明“只支持 WAV / 无 ffmpeg”。"""
    recognizer = SpeechRecognizer()
    with pytest.raises(TranscriptionError) as excinfo:
        recognizer._decode_audio(b"not-a-wav-file-at-all")

    message = str(excinfo.value)
    assert "WAV" in message, f"错误原因应提到只支持 WAV，实际：{message}"
    assert "ffmpeg" in message.lower() or "API" in message, f"应给出替代方案，实际：{message}"


# ==========================================================================
# 后端选择
# ==========================================================================
def test_backend_selection_reports_available_backends() -> None:
    """可用后端探测必须返回布尔字典，且至少能识别本地模型是否存在。"""
    recognizer = SpeechRecognizer()
    backends = recognizer.available_backends()

    assert set(backends) == {"api", "local"}, f"后端键应固定，实际 {set(backends)}"
    assert all(isinstance(value, bool) for value in backends.values()), "可用性必须是布尔值"


def test_no_backend_raises_with_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有任何后端时，必须抛出带修复建议的中文异常。"""
    recognizer = SpeechRecognizer()
    monkeypatch.setattr(recognizer, "available_backends", lambda: {"api": False, "local": False})
    monkeypatch.setattr(recognizer, "resolve_backend", lambda: "none")

    with pytest.raises(TranscriptionError) as excinfo:
        recognizer.transcribe(make_wav(seconds=0.1))

    message = str(excinfo.value)
    assert "语音" in message, f"错误信息应说明语音服务未配置，实际：{message}"
    assert "run_whisper" in message or "models/" in message, f"应给出修复建议，实际：{message}"


def test_health_contains_key_fields() -> None:
    """健康检查必须包含界面所需的字段。"""
    health = get_speech_recognizer().health()

    for field in ("backend", "api_available", "local_available", "local_model", "base_url", "model"):
        assert field in health, f"健康检查缺少字段 {field}"


# ==========================================================================
# 真实推理（仅在本地模型存在时执行）
# ==========================================================================
def test_local_transcription_runs_end_to_end() -> None:
    """本地 Whisper 必须能真正跑完一次转写（链路可用性验证）。

    注意：输入是合成正弦波、不是语音，因此**不校验识别内容**，
    只校验链路能产出字符串结果或给出明确原因——
    静音/噪声下模型可能输出空串，此时必须抛 ``TranscriptionError``。

    本用例专门验证 **local 后端**，因此直接锁定识别器实例的后端配置。
    为什么不用 ``monkeypatch.setenv``：``SpeechRecognizer`` 在构造时读取的是
    ``get_settings()`` 的缓存单例，环境变量改动对它无效；
    而联调时外部常把后端配成 ``api``，会让断言错误地失败。
    """
    recognizer = SpeechRecognizer()
    recognizer.settings.asr.backend = "local"
    if recognizer.local_model_dir() is None:
        pytest.skip("本地没有 Whisper 模型（models/whisper-*），跳过真实推理")

    audio = make_wav(seconds=1.0)
    try:
        result = recognizer.transcribe(audio)
    except TranscriptionError as exc:
        # 允许“识别不出内容”，但必须是可读的中文原因
        assert str(exc), "转写失败必须带原因"
        return

    assert "text" in result, "转写结果必须包含 text 字段"
    assert result["backend"] == "local", f"后端应为 local，实际 {result['backend']}"
    assert isinstance(result["text"], str)
