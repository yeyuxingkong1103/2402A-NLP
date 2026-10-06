# -*- coding: utf-8 -*-
"""
工单01 语音输入模块（问答界面的语音提问能力）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

支持两种语音识别引擎（自动探测、择优使用）：
  1. speech_recognition —— 麦克风录音 + Google Web Speech 在线识别，免费、免模型下载
  2. Whisper            —— openai-whisper 或 faster-whisper 本地离线识别，首次需下载模型

识别得到的文本可直接交给问答引擎（Pipeline）提问，也可被 qa_cli.py --voice 复用。

依赖安装（缺依赖时脚本会自动打印以下命令）：
    pip install SpeechRecognition pyaudio        # 在线识别（需联网）
    pip install openai-whisper                   # 离线识别（二选一）
    pip install faster-whisper                   # 离线识别，速度更快、内存更省
    # Windows 上 pyaudio 若无预编译轮子，可改用：
    #   pip install pipwin && pipwin install pyaudio

用法：
    python "工单01-基于PDF文档的问答系统/src/speech_input.py"            # 说一句，打印识别文本
    python .../src/speech_input.py --ask                                 # 识别后直接问答
    python .../src/speech_input.py --loop --ask                          # 连续语音问答（Ctrl+C 退出）
    python .../src/speech_input.py --file question.wav                   # 识别本地音频文件
    python .../src/speech_input.py --engine whisper --file q.wav --lang en-US
    python .../src/speech_input.py --list-mics                           # 列出可用麦克风
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

# --- 让脚本可以独立运行：把项目根目录（工单作业/）加入模块搜索路径 ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INSTALL_HELP = """缺少语音识别依赖，请按需安装：

  [在线识别，推荐先用这个]
      pip install SpeechRecognition pyaudio
      # Windows 上 pyaudio 编译失败时：
      #   pip install pipwin && pipwin install pyaudio

  [离线识别，可选，识别更稳但首次需下载模型]
      pip install openai-whisper      # 或
      pip install faster-whisper

  也可先用音频文件验证：python src/speech_input.py --file question.wav"""


# ---------------------------------------------------------------------------
# 依赖探测
# ---------------------------------------------------------------------------
def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def engine_status() -> dict:
    """返回各依赖的可用状态，用于选择识别引擎与打印环境报告。"""
    return {
        "speech_recognition": _module_available("speech_recognition"),
        "pyaudio": _module_available("pyaudio"),
        "whisper": _module_available("whisper"),
        "faster_whisper": _module_available("faster_whisper"),
    }


def describe_env() -> str:
    st = engine_status()
    def mark(ok: bool) -> str:
        return "已安装" if ok else "未安装"
    return (f"语音环境：speech_recognition={mark(st['speech_recognition'])} "
            f"pyaudio={mark(st['pyaudio'])} whisper={mark(st['whisper'])} "
            f"faster_whisper={mark(st['faster_whisper'])}")


def _lang_code(lang: str) -> str:
    """把 zh-CN / en-US 之类转成 Whisper 需要的语言码。"""
    return (lang or "zh-CN").split("-")[0].lower()


def _resolve_engine(engine: str) -> str:
    """决定实际使用的引擎：auto 时优先在线识别，其次本地 Whisper。"""
    st = engine_status()
    if engine != "auto":
        if engine == "sr" and not st["speech_recognition"]:
            raise RuntimeError("未安装 speech_recognition。\n" + INSTALL_HELP)
        if engine == "whisper" and not (st["whisper"] or st["faster_whisper"]):
            raise RuntimeError("未安装 whisper / faster-whisper。\n" + INSTALL_HELP)
        return engine
    if st["speech_recognition"]:
        return "sr"
    if st["whisper"] or st["faster_whisper"]:
        return "whisper"
    raise RuntimeError("没有可用的语音识别引擎。\n" + INSTALL_HELP)


# ---------------------------------------------------------------------------
# 录音（基于 speech_recognition / pyaudio）
# ---------------------------------------------------------------------------
def list_microphones() -> list[str]:
    if not _module_available("speech_recognition"):
        raise RuntimeError("需要 speech_recognition 才能枚举麦克风。\n" + INSTALL_HELP)
    import speech_recognition as sr
    try:
        return list(sr.Microphone.list_microphone_names())
    except Exception as e:
        raise RuntimeError(f"枚举麦克风失败：{e}（通常是未安装 pyaudio）\n{INSTALL_HELP}")


def _record_to_wav(path: Path, timeout: float, phrase_time_limit: float,
                   device_index: int | None) -> None:
    """用麦克风录一段话并写入 wav 文件（供本地 Whisper 离线识别使用）。"""
    try:
        import speech_recognition as sr
    except ImportError as e:
        raise RuntimeError(f"录音需要 speech_recognition：{e}\n{INSTALL_HELP}")

    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone(device_index=device_index)
    except Exception as e:
        raise RuntimeError(
            f"打开麦克风失败：{e}\n可能是未安装 pyaudio 或系统没有录音设备；"
            f"也可用 --file 指定音频文件。\n{INSTALL_HELP}")

    with mic as source:
        print(f"[录音] 请开始说话…（静音 {timeout:.0f} 秒后自动结束）")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout,
                                      phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            raise RuntimeError("没有检测到语音输入（超时）。")
    path.write_bytes(audio.get_wav_data())
    print(f"[录音] 结束，音频已保存：{path}")


# ---------------------------------------------------------------------------
# 识别：在线（Google Web Speech）
# ---------------------------------------------------------------------------
def _recognize_sr_file(path: Path, lang: str) -> str:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.AudioFile(str(path)) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio, language=lang or "zh-CN").strip()


def _recognize_sr_mic(lang: str, timeout: float, phrase_time_limit: float,
                      device_index: int | None) -> str:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone(device_index=device_index)
    except Exception as e:
        raise RuntimeError(
            f"打开麦克风失败：{e}\n可能是未安装 pyaudio 或系统没有录音设备；"
            f"也可用 --file 指定音频文件。\n{INSTALL_HELP}")

    with mic as source:
        print(f"[录音] 请开始说话…（静音 {timeout:.0f} 秒后自动结束）")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout,
                                      phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            raise RuntimeError("没有检测到语音输入（超时），请重试或调大 --timeout。")

    print("[识别] 正在调用在线语音识别…")
    try:
        return recognizer.recognize_google(audio, language=lang or "zh-CN").strip()
    except sr.UnknownValueError:
        raise RuntimeError("识别失败：没能听清内容，请靠近麦克风、放慢语速重说。")
    except sr.RequestError as e:
        raise RuntimeError(
            f"在线识别服务请求失败：{e}\n在线识别需要联网；"
            f"离线场景请改用 --engine whisper。")


# ---------------------------------------------------------------------------
# 识别：本地 Whisper
# ---------------------------------------------------------------------------
_whisper_cache: dict = {}


def _load_whisper(model_size: str):
    """加载并缓存 Whisper 模型（避免每次识别都重新加载）。"""
    if model_size in _whisper_cache:
        return _whisper_cache[model_size]
    if _module_available("whisper"):
        import whisper
        print(f"[模型] 加载 openai-whisper：{model_size} …")
        model = ("openai", whisper.load_model(model_size))
    elif _module_available("faster_whisper"):
        from faster_whisper import WhisperModel
        print(f"[模型] 加载 faster-whisper：{model_size}（CPU/int8）…")
        model = ("faster", WhisperModel(model_size, device="cpu",
                                        compute_type="int8"))
    else:
        raise RuntimeError("未安装 whisper / faster-whisper。\n" + INSTALL_HELP)
    _whisper_cache[model_size] = model
    return model


def _transcribe_whisper(path: Path, lang: str, model_size: str) -> str:
    backend, model = _load_whisper(model_size)
    code = _lang_code(lang)
    print("[识别] 本地 Whisper 识别中…")
    if backend == "openai":
        result = model.transcribe(str(path), language=code)
        return (result.get("text") or "").strip()
    segments, _info = model.transcribe(str(path), language=code)
    return "".join(seg.text for seg in segments).strip()


# ---------------------------------------------------------------------------
# 对外统一接口
# ---------------------------------------------------------------------------
def listen_once(lang: str = "zh-CN", timeout: float = 8.0,
                phrase_time_limit: float = 15.0, engine: str = "auto",
                device_index: int | None = None,
                whisper_model: str = "base") -> str | None:
    """
    从麦克风采集一句话并返回识别文本。

    Returns:
        识别文本；识别失败时返回 None（失败原因打印到终端，不抛异常）。
    """
    try:
        engine = _resolve_engine(engine)
        if engine == "whisper":
            # Whisper 本地识别需要音频文件：先用麦克风录 wav，再离线转写
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "speech.wav"
                _record_to_wav(wav, timeout, phrase_time_limit, device_index)
                text = _transcribe_whisper(wav, lang, whisper_model)
        else:
            text = _recognize_sr_mic(lang, timeout, phrase_time_limit, device_index)
    except RuntimeError as e:
        print(f"[语音] {e}")
        return None
    except Exception as e:                       # 兜底：任何异常都不应打断问答
        print(f"[语音] 识别过程出现异常：{e}")
        print(INSTALL_HELP)
        return None

    if not text:
        print("[语音] 未识别到有效内容，请重试。")
        return None
    return text


def recognize_file(path: str | Path, lang: str = "zh-CN", engine: str = "auto",
                   whisper_model: str = "base") -> str | None:
    """识别本地音频文件（wav/flac/aiff 等），失败返回 None。"""
    path = Path(path)
    if not path.exists():
        print(f"[语音] 音频文件不存在：{path}")
        return None
    try:
        engine = _resolve_engine(engine)
        text = (_transcribe_whisper(path, lang, whisper_model)
                if engine == "whisper" else _recognize_sr_file(path, lang))
    except RuntimeError as e:
        print(f"[语音] {e}")
        return None
    except Exception as e:
        print(f"[语音] 文件识别异常：{e}")
        return None
    return text or None


def ask(question: str, preset: str = "wo01_baseline", top_k: int | None = None) -> None:
    """把识别文本交给问答引擎（语音提问 -> RAG 回答）。"""
    from rag_core.pipeline import PRESETS, Pipeline

    pipeline = Pipeline(PRESETS[preset], collection="prospectus")
    try:
        pipeline.load_index()
    except Exception as e:
        print(f"[问答] 索引装载失败：{e}")
        print("       请先执行 build_index.py 建索引。")
        return
    if pipeline.retriever.vs.count() == 0:
        print("[问答] 向量索引为空，请先执行 build_index.py。")
        return

    print(f"\n[提问] {question}")
    try:
        result = pipeline.ask(question, top_k=top_k)
    except Exception as e:
        print(f"[问答] 生成失败：{e}")
        return
    print(f"[回答] {result.get('answer', '')}")
    for c in result.get("citations", []):
        print(f"       来源：《{c.get('doc', '')}》第{c.get('page', '?')}页")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="工单01：语音输入（支持麦克风/音频文件，在线或本地识别）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", default="auto", choices=["auto", "sr", "whisper"],
                   help="识别引擎：auto 自动选择（默认）/ sr 在线 / whisper 本地")
    p.add_argument("--lang", default="zh-CN",
                   help="识别语言，默认 zh-CN（英文用 en-US，Whisper 自动截取主语言码）")
    p.add_argument("--file", default=None, help="识别本地音频文件而非麦克风")
    p.add_argument("--ask", action="store_true", help="识别后直接调用问答引擎")
    p.add_argument("--loop", action="store_true", help="连续识别（Ctrl+C 退出）")
    p.add_argument("--timeout", type=float, default=8.0, help="静音超时秒数，默认 8")
    p.add_argument("--phrase-time-limit", type=float, default=15.0,
                   help="单句最长录音秒数，默认 15")
    p.add_argument("--device-index", type=int, default=None, help="麦克风设备序号")
    p.add_argument("--list-mics", action="store_true", help="列出可用麦克风后退出")
    p.add_argument("--whisper-model", default="base",
                   help="Whisper 模型规格（tiny/base/small/medium/large），默认 base")
    p.add_argument("--preset", default="wo01_baseline", help="--ask 时使用的流水线预设")
    p.add_argument("--top-k", type=int, default=None, help="--ask 时的检索片段数")
    return p


def main() -> int:
    args = build_parser().parse_args()
    print("=" * 66)
    print("  工单01 语音输入：基于PDF文档的问答系统")
    print("  " + describe_env())
    print("=" * 66)

    if args.list_mics:
        try:
            for i, name in enumerate(list_microphones()):
                print(f"  [{i}] {name}")
        except RuntimeError as e:
            print(f"[错误] {e}")
            return 2
        return 0

    if args.file:
        text = recognize_file(args.file, lang=args.lang, engine=args.engine,
                              whisper_model=args.whisper_model)
        if not text:
            return 3
        print(f"\n[识别结果] {text}")
        if args.ask:
            ask(text, preset=args.preset, top_k=args.top_k)
        return 0

    # 麦克风模式
    try:
        _resolve_engine(args.engine)
    except RuntimeError as e:
        print(f"[错误] {e}")
        return 2

    while True:
        text = listen_once(lang=args.lang, timeout=args.timeout,
                           phrase_time_limit=args.phrase_time_limit,
                           engine=args.engine, device_index=args.device_index,
                           whisper_model=args.whisper_model)
        if text:
            print(f"\n[识别结果] {text}")
            if args.ask:
                ask(text, preset=args.preset, top_k=args.top_k)
        if not args.loop:
            return 0 if text else 3
        print("\n（--loop 模式：继续识别，Ctrl+C 退出）")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已退出。")
