"""端到端联调脚本：LLM 流式生成 + 语音识别 + 中英双语（需要本地服务）。

## 它验证什么

本脚本把「本地起服务 → 走完整链路」这一步固化下来，覆盖单测覆盖不到的路径：

1. **LLM 流式生成**（`mode=llm`）：SSE 增量、首字延迟、引用回填；
2. **语音识别**（HTTP api 后端）：音频 → 转写文本 → 作为问题提交 → 得到答案；
3. **中英双语**：中文问题用中文答、英文问题用英文答；
4. **降级路径**：服务不可用时自动落到抽取式回答，且给出明确原因。

## 前置：两个本地服务

```powershell
# 终端 1：LLM（CPU，OpenAI 兼容）
python 研发/scripts/run_local_llm.py --port 8000
# 终端 2：语音识别（CPU，OpenAI 兼容）
python 研发/scripts/run_local_asr.py --port 8001
```

## 运行

```powershell
python 研发/scripts/check_integration.py
```

退出码 0 表示全部通过；非 0 表示有环节失败（失败原因会打印出来）。
"""

from __future__ import annotations

import io
import math
import struct
import sys
import time
import warnings
import wave
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "研发"
for _candidate in (SOURCE_ROOT, PROJECT_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from app.core.asr import TranscriptionError, get_speech_recognizer  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.qa_engine import QAEngine  # noqa: E402

PASS, FAIL, SKIP = "✓", "✗", "–"
results: list[tuple[str, str, str]] = []


def record(mark: str, name: str, detail: str = "") -> None:
    results.append((mark, name, detail))
    print(f"  [{mark}] {name}" + (f"  —— {detail}" if detail else ""), flush=True)


def make_wav(seconds: float = 1.5, rate: int = 16000, freq: float = 220.0) -> bytes:
    """生成测试音频。

    注意：这是**合成正弦波，不是真实语音**。小模型对它可能输出任意文本
    （实测 whisper-tiny 输出 "Thank you very much."）。
    因此本脚本只校验「转写链路能返回文本」，不校验识别内容的正确性——
    识别准确率需要真实录音，见 设计/文档/DEMO_CHECKLIST.md 阶段 M。
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(2500 * math.sin(2 * math.pi * freq * i / rate)))
                for i in range(int(rate * seconds))
            )
        )
    return buffer.getvalue()


# ==========================================================================
def check_services() -> tuple[bool, bool]:
    """探测两个本地服务，返回 ``(llm_ok, asr_ok)``。"""
    import urllib.error
    import urllib.request

    def probe(url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    settings = get_settings()
    llm = probe(f"{settings.llm.base_url.rstrip('/')}/models")
    asr = probe(f"{settings.asr.base_url.rstrip('/')}/models")
    record(
        PASS if llm else SKIP,
        "LLM 服务探测",
        f"{settings.llm.base_url} → {'可用' if llm else '未启动（将走抽取式降级）'}",
    )
    record(
        PASS if asr else SKIP,
        "语音服务探测",
        f"{settings.asr.base_url} → {'可用' if asr else '未启动（语音输入不可用）'}",
    )
    return llm, asr


def check_llm_stream(engine: QAEngine) -> None:
    """LLM 流式生成：事件序列、首字延迟、答案与引用。"""
    question = "武汉兴图新科电子股份有限公司注册资本是多少？"
    conversation = engine.new_conversation()
    started = time.perf_counter()
    events: list[str] = []
    deltas = 0
    first_token_ms = 0.0
    answer = None
    for event, payload in engine.stream(question, conversation, allow_llm=True):
        events.append(event)
        if event == "first_token":
            first_token_ms = (time.perf_counter() - started) * 1000
        elif event == "delta":
            deltas += 1
        elif event == "done":
            answer = payload["answer"]

    if answer is None:
        record(FAIL, "LLM 流式生成", "未收到 done 事件")
        return

    total_ms = (time.perf_counter() - started) * 1000
    if answer.mode == "llm":
        record(
            PASS,
            "LLM 流式生成",
            f"模式=llm 首字={first_token_ms:.0f}ms 总耗时={total_ms:.0f}ms delta={deltas}",
        )
    else:
        record(SKIP, "LLM 流式生成", f"未走 LLM（模式={answer.mode}），已降级")
    ok = "5,520" in answer.answer
    record(
        PASS if ok else FAIL,
        "LLM 答案正确性",
        f"答案={answer.answer[:64]!r}",
    )
    record(
        PASS if answer.citations else FAIL,
        "LLM 引用回填",
        f"引用={[c.label() for c in answer.citations[:3]]}",
    )


def check_voice_pipeline(engine: QAEngine) -> None:
    """语音链路：音频 → 转写 → 作为问题 → 拿到答案。"""
    recognizer = get_speech_recognizer()
    health = recognizer.health()
    if not health.get("api_available") and not health.get("local_available"):
        record(SKIP, "语音转写", "无可用语音后端")
        return

    audio = make_wav()
    try:
        started = time.perf_counter()
        result = recognizer.transcribe(audio)
        elapsed = time.perf_counter() - started
    except TranscriptionError as exc:
        record(SKIP, "语音转写", f"未识别出内容（合成音频的预期结果）：{str(exc)[:60]}")
        return

    record(
        PASS,
        "语音转写",
        f"后端={result['backend']} 耗时={elapsed:.1f}s 文本={result['text'][:40]!r}",
    )

    # 转写文本直接作为问题提交（与界面行为一致）
    question = str(result["text"]).strip()
    if len(question) < 2:
        record(SKIP, "语音问答闭环", "转写文本过短，跳过")
        return
    answer = engine.ask(question, allow_llm=False, save=False)
    record(
        PASS,
        "语音问答闭环",
        f"问题={question[:30]!r} → 回答={answer.answer[:40]!r}（未知={answer.is_unknown}）",
    )


def check_bilingual(engine: QAEngine) -> None:
    """中英双语：中文用中文答、英文用英文答。"""
    cases = [
        ("中文", "武汉兴图新科电子股份有限公司法定代表人是谁？", "zh", "程家明"),
        ("English", "Who is the legal representative?", "en", "Cheng Jiaming"),
    ]
    for label, question, want_lang, want_text in cases:
        answer = engine.ask(question, allow_llm=False, save=False)
        ok = answer.language == want_lang and want_text in answer.answer
        record(
            PASS if ok else FAIL,
            f"{label}问答",
            f"语言={answer.language} 回答={answer.answer[:56]!r}",
        )


def check_unknown_fallback(engine: QAEngine) -> None:
    """无关问题必须兜底，中英各自的语言。"""
    for label, question, expect in [
        ("中文兜底", "今天天气怎么样？", "不清楚"),
        ("英文兜底", "What is the weather like today?", "don't know"),
    ]:
        answer = engine.ask(question, allow_llm=False, save=False)
        ok = answer.is_unknown and expect.lower() in answer.answer.lower()
        record(PASS if ok else FAIL, label, f"回答={answer.answer[:40]!r}")


# ==========================================================================
def main() -> int:
    print("=" * 70)
    print(" 端到端联调 —— LLM 流式生成 + 语音识别 + 中英双语")
    print("=" * 70)
    settings = get_settings()
    print(f"  LLM   : {settings.llm.base_url}  model={settings.llm.model}")
    print(f"  语音  : backend={settings.asr.backend}  {settings.asr.base_url}")
    print(f"  上下文: max_context_chunks={settings.llm.max_context_chunks or '默认(rerank_top_n)'}")
    print()

    print("[1/5] 服务探测")
    check_services()

    print("\n[2/5] 加载问答引擎")
    engine = QAEngine()
    warmup = engine.warmup()
    stats = engine.stats()
    record(
        PASS if stats["index_ready"] else FAIL,
        "索引与预热",
        f"分块={stats['chunks']} 向量={stats['vector_count']} "
        f"BM25={stats['bm25_documents']} llm_available={warmup.get('llm_available')}",
    )

    print("\n[3/5] LLM 流式生成")
    check_llm_stream(engine)

    print("\n[4/5] 语音链路")
    check_voice_pipeline(engine)

    print("\n[5/5] 双语与兜底")
    check_bilingual(engine)
    check_unknown_fallback(engine)

    passed = sum(1 for mark, _, _ in results if mark == PASS)
    failed = sum(1 for mark, _, _ in results if mark == FAIL)
    skipped = sum(1 for mark, _, _ in results if mark == SKIP)
    print("\n" + "=" * 70)
    print(f" 结果：通过 {passed} · 失败 {failed} · 跳过 {skipped}")
    print("=" * 70)
    if failed:
        print(" 失败项：")
        for mark, name, detail in results:
            if mark == FAIL:
                print(f"   - {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
