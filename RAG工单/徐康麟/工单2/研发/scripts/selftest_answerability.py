"""可答性稳定性自测：无关问题集连跑 3 次，每次均须拒答（t13 验收 1）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 自测脚本（t13 证据；正式判分由 T5/T9 负责）

背景（t13）：含公司名/文档名的无关问题在"实词覆盖度"上天然很高，会被误判可答；
tester 实测同一批 12 条样本两次失败题不同 → **稳定性本身是验收项**（连跑 3 次均须 12/12）。

同时输出**正常问题对照**（中文 10 题判对数 + 英文 5 题作答数），避免"为过测而过度收紧"。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability.py                    # 抽取式引擎（确定性）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability.py --engine production  # 与服务端同构

``--engine production`` 用 ``get_qa_engine()``——与 ``app/ui/serve_fallback.py`` **同一个引擎配置**
（LLM 可用时走 LLM，不可用自动降级）。为什么需要：可答性闸门位于生成之前，
但验收 4 的正式判分走的是**在线服务路径**，用同一引擎复跑三次才能证明三次结果可代表线上行为。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.evaluator import get_evaluator  # noqa: E402
from app.core.qa_engine import QAEngine, get_qa_engine  # noqa: E402
from app.models.schemas import GoldenQA  # noqa: E402

TEST_DATA = Path(__file__).resolve().parents[2] / "测试" / "测试数据"
UNKNOWN_FILE = TEST_DATA / "unknown_questions.jsonl"
GOLDEN_FILE = TEST_DATA / "golden_qa.jsonl"
EN_QUESTIONS = (
    "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
    "Who is the legal representative of the company?",
    "How much of the raised funds will be used to supplement working capital?",
    "Which technical standard did the company participate in formulating?",
    "What percentage of main business revenue came from the military sector during the reporting period?",
)
#: 多轮追问回归点（t1）：第 3 轮曾被闸门误拒（``rare_term_missing:哪里``）。
MULTI_TURNS: tuple[tuple[str, str], ...] = (
    ("武汉兴图新科电子股份有限公司法定代表人是谁？", "第1轮·直问"),
    ("那它的注册资本呢？", "第2轮·指代追问"),
    ("它的注册地址在哪里？", "第3轮·继续追问"),
)
#: 扩展无关集（同类泛化）：与官方 12 条同类但不同题，用于证明修复是"规则"而非"逐题特判"。
#: 前 3 条来自 captain 独立复核（首条在闸门 v1 上曾误答成合同叙述 [页码: 57]）。
EXTRA_UNKNOWN: tuple[str, ...] = (
    "帮我写一首关于股票的诗",
    "帮我写一段公司简介",
    "预测一下公司明年利润",
    "帮我编一个关于公司的笑话",
    "你觉得公司股票值得买吗",
    "写一首关于春天的诗",
)
UNKNOWN_TEXT = "不清楚"
ROUNDS = 3


def main() -> int:
    """自测入口；退出码 0 = 全部通过。"""
    use_production = "--engine" in sys.argv and "production" in sys.argv
    if use_production:
        engine: QAEngine = get_qa_engine()
        engine_label = "production（get_qa_engine：与 serve_fallback.py 同构）"
    else:
        engine = QAEngine(force_extractive=True)
        engine_label = "extractive（QAEngine(force_extractive=True)）"
    print(f"引擎：{engine_label}")
    if not engine.load_index():
        print("业务失败：索引未就绪，请先执行 研发/scripts/build_index.py", file=sys.stderr)
        return 2
    engine.warmup()
    unknown = [
        json.loads(line) for line in UNKNOWN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    golden = [GoldenQA(**json.loads(line)) for line in GOLDEN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluator = get_evaluator(engine._store)  # noqa: SLF001

    round_results: list[list[bool]] = []
    extra_results: list[list[bool]] = []
    for run in range(1, ROUNDS + 1):
        flags: list[bool] = []
        print("=" * 96)
        print(f"第 {run}/{ROUNDS} 次运行（官方无关问题 {len(unknown)} 条 + 扩展同类 {len(EXTRA_UNKNOWN)} 条）")
        for item in unknown:
            answer = engine.ask(item["question"])
            ok = bool(answer.is_unknown) and answer.answer.strip() == UNKNOWN_TEXT
            flags.append(ok)
            print(
                f"  [{'✅' if ok else '❌'}] {item['question'][:34]:<36} → "
                f"is_unknown={answer.is_unknown} text={answer.answer[:26]!r}"
            )
        round_results.append(flags)
        extra_flags: list[bool] = []
        for question in EXTRA_UNKNOWN:
            answer = engine.ask(question)
            ok = bool(answer.is_unknown) and answer.answer.strip() == UNKNOWN_TEXT
            extra_flags.append(ok)
            print(
                f"  [{'✅' if ok else '❌'}] {question[:34]:<36} → "
                f"is_unknown={answer.is_unknown} text={answer.answer[:26]!r}（扩展同类）"
            )
        extra_results.append(extra_flags)
        print(f"  —— 第 {run} 次：官方 {sum(flags)}/{len(flags)}；扩展同类 {sum(extra_flags)}/{len(extra_flags)}")

    print("=" * 96)
    print("正常问题对照（同一引擎、同一次运行）")
    correct = 0
    for item in golden:
        answer = engine.ask(item.question)
        record = evaluator.evaluate_answer(item, answer, mode="extractive")
        correct += 1 if record.is_correct else 0
    en_answered = 0
    for question in EN_QUESTIONS:
        answer = engine.ask(question)
        if not answer.is_unknown and answer.citations and any(ch.isascii() and ch.isalpha() for ch in answer.answer):
            en_answered += 1
    print(f"  中文 10 题判对：{correct}/10；英文 5 题作答：{en_answered}/5")

    # 多轮追问：同会话顺序提问，三轮**都必须作答**（t1 回归点：曾被 rare_term_missing:哪里 误拒）。
    # 为什么必须放进自测：只覆盖 10 中文 + 5 英文时，"指代追问"从来没被校验过，
    # 于是"闸门把合法追问判死"能一路漏到在线套件才暴露（工单2 t1 实际发生过）。
    print("=" * 96)
    print("多轮追问对照（同一引擎、同一会话）")
    multi_ok = 0
    conversation_id = engine.new_conversation("selftest-多轮追问")
    for question, label in MULTI_TURNS:
        answer = engine.ask(question, conversation_id=conversation_id)
        text = (answer.answer or "").strip()
        ok = (not answer.is_unknown) and bool(text)
        multi_ok += 1 if ok else 0
        print(f"  [{'✅作答' if ok else '❌误拒'}] {label}：{question!r} → {text[:40]!r} ({answer.mode})")
    print(f"  多轮作答：{multi_ok}/{len(MULTI_TURNS)}")

    stable = all(all(flags) for flags in round_results)
    extra_stable = all(all(flags) for flags in extra_results)
    ok_all = (
        stable
        and extra_stable
        and correct >= 9
        and en_answered == 5
        and multi_ok == len(MULTI_TURNS)
    )
    print("=" * 96)
    print(f"三次拒答结果（官方）：{[sum(flags) for flags in round_results]}/{len(unknown)}"
          f"（每次均须 {len(unknown)}/{len(unknown)}）")
    print(f"三次拒答结果（扩展同类）：{[sum(flags) for flags in extra_results]}/{len(EXTRA_UNKNOWN)}"
          f"（每次均须 {len(EXTRA_UNKNOWN)}/{len(EXTRA_UNKNOWN)}）")
    print(f"结论：{'PASS ✅' if ok_all else 'FAIL ❌'}"
          f"（稳定性={'满足' if stable and extra_stable else '不满足'}；中文 {correct}/10 ≥9；"
          f"英文 {en_answered}/5；多轮 {multi_ok}/{len(MULTI_TURNS)}）")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
