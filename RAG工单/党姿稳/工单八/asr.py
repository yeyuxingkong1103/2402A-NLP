# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
语音输入（ASR）模块：把界面录制的音频转写为文字问题。
优先使用本地 SpeechRecognition + Google Web 识别；不可用时返回 None 并由界面提示改用手动输入。
"""
import os
import tempfile


def transcribe(audio_bytes, language="zh-CN"):
    """
    audio_bytes: 录音原始字节（WAV）
    返回识别文本；失败返回 None
    """
    if not audio_bytes:
        return None
    try:
        import speech_recognition as sr
    except ImportError:
        print("[ASR] 未安装 speech_recognition，语音输入不可用（可 pip install SpeechRecognition）")
        return None
    tmp = os.path.join(tempfile.gettempdir(), "rag_voice.wav")
    with open(tmp, "wb") as f:
        f.write(audio_bytes)
    r = sr.Recognizer()
    try:
        with sr.AudioFile(tmp) as src:
            audio = r.record(src)
        return r.recognize_google(audio, language=language)
    except Exception as e:
        print(f"[ASR] 识别失败: {repr(e)[:120]}")
        return None
