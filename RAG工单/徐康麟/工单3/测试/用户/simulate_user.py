# -*- coding: utf-8 -*-
"""用户级：模拟用户旅程脚本（验收 9 的第三条命令）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用法（工作目录 = E:\\gao6gongdan\\工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/用户/simulate_user.py

行为：按「真实用户会怎么用」依次跑 9 个场景（首次提问 → 追问 → 切语料 → 英文 →
语料外拒答 → 同页互污两题 → 编造红线 → 闸门 fail-open），逐场景打印问答与引用，
把结果落盘到 ``测试/留痕/simulate_user.txt`` 与 ``simulate_user.json``，
全部通过退出码 0，否则 1。

报告纪律：输出首行固定为「RAGAS 未运行（依赖不可用，本机断网）」；不许出现 RAGAS 数值。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent
for extra in (str(TESTS_DIR), str(REPO_ROOT / "研发")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from common import assertions, golden as golden_mod, paths  # noqa: E402
from common.reports import assert_no_ragas_numbers, write_report, write_text  # noqa: E402


def _citation_text(answer: Any) -> str:
    """把答案里的引用渲染成可读文本。"""
    citations = list(getattr(answer, "citations", []) or [])
    if not citations:
        return "（无引用）"
    out: list[str] = []
    for cite in citations:
        render = getattr(cite, "render", None)
        out.append(render() if callable(render) else f"[{getattr(cite, 'file_name', '?')}: {getattr(cite, 'page', '?')}]")
    return "、".join(out)


def _scenario_defs(golden: dict[int, Any]) -> list[dict[str, Any]]:
    """场景定义（每个场景给出提问、可选过滤与判定函数）。"""
    pdf2_name = getattr(golden_mod.corpus_path(golden[1]), "name", "")  # 自动发现，禁硬编码文件名

    def check_numeric(answer: Any, item: Any) -> list[assertions.Check]:
        """数值题：答案要素齐备 + 首字预算 + 引用非空。"""
        checks = assertions.check_leakage(str(answer.text), required=item.required_substrings,
                                          forbidden=item.forbidden_substrings, label=f"题 {item.id}")
        checks.append(assertions.Check(f"首字 ≤3000ms（题 {item.id}）",
                                       0 < float(getattr(answer, "first_token_ms", 0.0) or 0.0) <= 3000,
                                       f"{getattr(answer, 'first_token_ms', None)} ms"))
        checks.append(assertions.Check(f"带引用（题 {item.id}）", bool(getattr(answer, "citations", [])),
                                       _citation_text(answer)))
        return checks

    def check_unknown(answer: Any) -> list[assertions.Check]:
        """语料外问题：必须拒答且无引用。"""
        return assertions.check_unknown_answer(answer, label="语料外")

    def check_followup(answer: Any) -> list[assertions.Check]:
        """多轮追问：指代消解必须落到程家明。"""
        return [assertions.Check("追问消解 → 程家明", "程家明" in str(answer.text),
                                 f"答案：{str(answer.text)[:50]!r}")]

    def check_english(answer: Any) -> list[assertions.Check]:
        """英文提问：语言标注 en 且带引用。"""
        return [
            assertions.Check("英文提问 language == 'en'",
                             str(getattr(answer, "language", "")) == "en",
                             f"language={getattr(answer, 'language', None)!r}"),
            assertions.Check("英文答案带引用", bool(getattr(answer, "citations", [])), _citation_text(answer)),
        ]

    def check_no_fabrication(answer: Any) -> list[assertions.Check]:
        """编造红线：不得输出 15,000，且必须拒答。"""
        checks = assertions.check_unknown_answer(answer, label="PDF2 补流")
        checks += assertions.check_no_fabrication(answer, forbidden_substrings=["15,000", "15000"],
                                                 label="PDF2 补流")
        return checks

    def check_gate(answer: Any, question_id: int) -> list[assertions.Check]:
        """闸门 fail-open：不得拒答；有 subject_gate 时必须 ok=True。"""
        checks = [assertions.Check(f"题 {question_id} 不得拒答", not bool(getattr(answer, "is_unknown", False)),
                                   f"is_unknown={getattr(answer, 'is_unknown', None)}")]
        gate = getattr(answer, "subject_gate", None)
        if gate is not None:
            checks += assertions.check_subject_gate_fail_open(gate, label=f"题 {question_id}")
        return checks

    return [
        {"id": "U1", "title": "首次提问：注册资本",
         "question": golden[543].question, "call": lambda engine: engine.ask(
             golden[543].question, session_id="sim-user", stream=False),
         "check": lambda answer: check_numeric(answer, golden[543])},
        {"id": "U2", "title": "多轮追问：那法定代表人呢？",
         "question": "那法定代表人呢？", "call": lambda engine: engine.ask(
             "那法定代表人呢？", session_id="sim-user", stream=False),
         "check": check_followup},
        {"id": "U3", "title": "切换语料：PDF2 发行股数",
         "question": golden[1].question, "file_names": pdf2_name,
         "call": lambda engine: engine.ask(golden[1].question, session_id="sim-pdf2",
                                           file_names=[pdf2_name] if pdf2_name else None,
                                           stream=False),
         "check": lambda answer: check_numeric(answer, golden[1])},
        {"id": "U4", "title": "英文提问：registered capital",
         "question": "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
         "call": lambda engine: engine.ask(
             "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
             session_id="sim-en", stream=False),
         "check": check_english},
        {"id": "U5", "title": "语料外提问：必须回「不清楚」",
         "question": "银河系外文明的量子计算机专利申请数量是多少？",
         "call": lambda engine: engine.ask("银河系外文明的量子计算机专利申请数量是多少？",
                                           session_id="sim-unknown", stream=False),
         "check": check_unknown},
        {"id": "U6", "title": "编造红线：问 PDF2 补流（必须拒答，不得给 15,000）",
         "question": "根据武汉力源信息技术股份有限公司招股意向书，本次发行募集资金中有多少用于补充流动资金？",
         "call": lambda engine: engine.ask(
             "根据武汉力源信息技术股份有限公司招股意向书，本次发行募集资金中有多少用于补充流动资金？",
             session_id="sim-n4", file_names=[pdf2_name] if pdf2_name else None, stream=False),
         "check": check_no_fabrication},
        {"id": "U7", "title": "同页互污：题 4 关联方企业（不得出现自然人赵马克）",
         "question": golden[4].question,
         "call": lambda engine: engine.ask(golden[4].question, session_id="sim-leak4", stream=False),
         "check": lambda answer: assertions.check_leakage(
             str(answer.text), required=golden[4].required_substrings,
             forbidden=golden[4].forbidden_substrings, label="题 4")},
        {"id": "U8", "title": "同页互污：题 3 控股股东（不得夹带 7 家企业）",
         "question": golden[3].question,
         "call": lambda engine: engine.ask(golden[3].question, session_id="sim-leak3", stream=False),
         "check": lambda answer: assertions.check_leakage(
             str(answer.text), required=golden[3].required_substrings,
             forbidden=golden[3].forbidden_substrings, label="题 3")},
        {"id": "U9", "title": "闸门 fail-open：题 34 / 793 不得被判不可答",
         "question": golden[34].question,
         "call": lambda engine: engine.ask(golden[34].question, session_id="sim-gate34", stream=False),
         "check": lambda answer: check_gate(answer, 34)},
    ]


def main() -> int:
    """跑完用户旅程，输出报告并返回退出码。"""
    from app.core.config import get_config  # noqa: PLC0415
    from app.core.logging_conf import setup_logging  # noqa: PLC0415
    from app.core.qa_engine import build_engine  # noqa: PLC0415

    golden = golden_mod.by_id(golden_mod.load_golden())
    failures: list[str] = []
    lines: list[str] = [assertions.RAGAS_BANNER,
                        f"工单：{assertions.WORK_ORDER}",
                        "场景：模拟用户旅程（离线/在线/用户 三级中的用户级）",
                        "=" * 78]

    try:
        setup_logging(get_config(), force=True)
        engine = build_engine(warmup=True)
    except Exception as exc:  # noqa: BLE001 —— 引擎起不来必须显式失败，不静默
        import traceback

        lines.append(f"❌ 问答引擎启动失败：{type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
        body = "\n".join(lines)
        print(body)
        write_text("simulate_user.txt", body)
        return 1

    rows: list[dict[str, Any]] = []
    for scenario in _scenario_defs(golden):
        started = time.perf_counter()
        try:
            answer = scenario["call"](engine)
            checks = scenario["check"](answer)
        except Exception as exc:  # noqa: BLE001 —— 单场景异常不能让整轮静默跳过
            import traceback

            checks = [assertions.Check(f"场景执行异常（{scenario['id']}）", False,
                                       f"{type(exc).__name__}: {exc}")]
            lines.append(traceback.format_exc())
            answer = None

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        bad = [c for c in checks if not c.ok]
        rows.append({
            "id": scenario["id"], "title": scenario["title"], "question": scenario["question"],
            "answer_text": str(getattr(answer, "text", "") or ""),
            "is_unknown": bool(getattr(answer, "is_unknown", False)),
            "unknown_reason": getattr(answer, "unknown_reason", None),
            "citations": _citation_text(answer) if answer is not None else "",
            "first_token_ms": float(getattr(answer, "first_token_ms", 0.0) or 0.0),
            "wall_ms": elapsed_ms,
            "checks": [c.to_dict() for c in checks],
            "ok": not bad,
        })

        lines.append(f"\n[{scenario['id']}] {scenario['title']}")
        lines.append(f"  Q：{scenario['question']}")
        lines.append(f"  A：{rows[-1]['answer_text'][:300]}")
        lines.append(f"  引用：{rows[-1]['citations']}　首字：{rows[-1]['first_token_ms']} ms　耗时：{elapsed_ms} ms")
        for check in checks:
            lines.append("  " + check.render())
        if bad:
            failures.append(f"[{scenario['id']}] " + "；".join(c.render() for c in bad))

    passed = sum(1 for row in rows if row["ok"])
    lines.append("\n" + "=" * 78)
    lines.append(f"用户旅程合计：✅{passed} / {len(rows)}　❌{len(rows) - passed}")
    lines.append(f"注：{assertions.RAGAS_BANNER}")
    body = "\n".join(lines)
    print(body)

    write_report("simulate_user", "模拟用户旅程", rows,
                 extra={"passed": passed, "total": len(rows)})
    write_text("simulate_user.txt", body)
    assert_no_ragas_numbers(rows)
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底必须打印堆栈
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
