# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
语音输入：后端 ASR。

【为什么不用浏览器的 Web Speech API】
Chrome 的 SpeechRecognition 把音频发到 **Google 的服务器**做识别。
国内网络不通，且失败时它只返回一个笼统的 `network` 错误 ——
表现是「点麦克风没反应」或「提示麦克风坏了」，
让人误以为是设备问题，实际是网络问题。排查成本极高。
因此改为：前端 MediaRecorder 录音 → POST 到后端 → 本地 faster-whisper 识别。

【可降级】faster-whisper 在 requirements-asr.txt 里单独安装（工单01 里
它是唯一可砍的功能，装不上不影响其余验收项）。未安装时本接口返回 503
并说明安装方式，前端据此禁用录音按钮，而不是整个服务起不来。
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings

router = APIRouter(tags=["asr"])

_model = None
_model_lock = asyncio.Lock()

INSTALL_HINT = (
    "语音功能未安装。安装方式：\n"
    "  pip install -r requirements-asr.txt\n"
    "  模型权重需从镜像下载（huggingface.co 国内不通）：\n"
    "  HF_ENDPOINT=https://hf-mirror.com python -c "
    "\"from faster_whisper import WhisperModel; WhisperModel('small')\""
)


async def _get_model():
    global _model
    async with _model_lock:
        if _model is not None:
            return _model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise HTTPException(503, INSTALL_HINT) from e

        # CPU int8：不占 8GB 显存（显存已被 qwen3 + bge-m3 占满），
        # small 模型 CPU int8 上识别一句话约 1–2 秒，可接受。
        def _load():
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            return WhisperModel("small", device="cpu", compute_type="int8")

        try:
            _model = await asyncio.to_thread(_load)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"ASR 模型加载失败：{e}\n{INSTALL_HINT}") from e
        return _model


@router.get("/api/asr/available")
async def asr_available() -> dict:
    try:
        import faster_whisper  # noqa: F401
        return {"available": True, "backend": "faster-whisper(small, cpu, int8)",
                "loaded": _model is not None}
    except ImportError:
        return {"available": False, "hint": INSTALL_HINT}


@router.post("/api/asr")
async def transcribe(file: UploadFile = File(...), language: str | None = None) -> dict:
    """接收浏览器录音（webm/ogg/wav），返回识别文本。"""
    model = await _get_model()

    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    data = await file.read()
    if not data:
        raise HTTPException(400, "音频为空")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()

        def _run():
            # vad_filter 去掉静音段，能显著降低短语音的误识别
            segments, info = model.transcribe(
                tmp.name, language=language, vad_filter=True, beam_size=5,
            )
            return "".join(s.text for s in segments).strip(), info

        text, info = await asyncio.to_thread(_run)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return {
        "text": text,
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
    }
