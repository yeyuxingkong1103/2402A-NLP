# -*- coding: utf-8 -*-
"""用户级：不可答负例、同页互污染（N-5）、闸门 fail-open（N-6）、编造红线（N-4）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用户视角关心的是「会不会被误导」：
    * 问语料里没有的东西 → 必须回「不清楚」，且不得编造金额；
    * 题 3 / 题 4 证据同页（PDF2 物理 157）→ 不得互相夹带；
    * 题 34/793 的证据是正文段 → 闸门必须 fail-open，不得误判不可答；
    * 题 207 的金额不得被 PDF2 物理 340 的银行借款污染（红线⑨）。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import assertions
from common.reports import write_report

pytestmark = [pytest.mark.user, pytest.mark.slow]


def _ask(engine: Any, question: str, *, session: str, file_names: list[str] | None = None) -> Any:
    """统一调用入口（``file_names`` 为空则不过滤）。"""
    return engine.ask(question, session_id=session, file_names=file_names or None, stream=False)


def test_unknown_questions_refused(engine: Any, unknown_cases: list[Any]) -> None:
    """全部不可答负例必须回「不清楚」、无引用、且不含被禁止的编造内容。"""
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for case in unknown_cases:
        answer = _ask(engine, case.question, session=f"t8-unk-{case.id}", file_names=case.file_names)
        checks = assertions.check_unknown_answer(answer, label=case.id)
        checks += assertions.check_no_fabrication(
            answer, forbidden_substrings=case.must_not_contain, label=case.id)
        bad = [c for c in checks if not c.ok]
        rows.append({"case": case.id, "question": case.question, "text": str(answer.text),
                     "is_unknown": bool(getattr(answer, "is_unknown", False)),
                     "unknown_reason": getattr(answer, "unknown_reason", None),
                     "citations": [getattr(c, "render", lambda: str(c))() for c in
                                   (getattr(answer, "citations", []) or [])],
                     "checks": [c.to_dict() for c in checks]})
        if bad:
            failures.append(f"[{case.id}] " + "；".join(c.render() for c in bad))
    write_report("user_unknown_cases", "不可答负例执行结果", rows)
    assert not failures, ("不可答负例未达标（应回「不清楚」而实际作答）：\n" + "\n".join(failures)
                          + "\n【性质：产品缺陷 —— 拒答机制失效（含 N-4 编造红线：把 PDF2 银行借款"
                            "答成 IPO 补流），影响验收 6；captain 已并入 engineer 修单】")


def test_leakage_and_element_rules(engine: Any, leakage_cases: list[Any]) -> None:
    """互污染/要素齐备用例：required 必含、forbidden 必不含。"""
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for case in leakage_cases:
        answer = _ask(engine, case.question, session=f"t8-leak-{case.id}")
        checks = assertions.check_leakage(str(answer.text),
                                          forbidden=case.forbidden_substrings,
                                          required=case.required_substrings, label=case.id)
        if case.expect_unknown:
            checks += assertions.check_unknown_answer(answer, label=case.id)
        else:
            # 可答用例**不得被拒答**（题 4 曾被闸门以 subject_gate:org_leaked 误拒答）
            checks.append(assertions.Check(
                f"不得被拒答（{case.id}）", not bool(getattr(answer, "is_unknown", False)),
                f"is_unknown={getattr(answer, 'is_unknown', None)}　"
                f"reason={getattr(answer, 'unknown_reason', None)}"))
        bad = [c for c in checks if not c.ok]
        rows.append({"case": case.id, "question": case.question, "text": str(answer.text),
                     "is_unknown": bool(getattr(answer, "is_unknown", False)),
                     "unknown_reason": getattr(answer, "unknown_reason", None),
                     "checks": [c.to_dict() for c in checks]})
        if bad:
            failures.append(f"[{case.id}] " + "；".join(c.render() for c in bad))
    write_report("user_leakage_cases", "互污染与要素齐备用例执行结果", rows)
    assert not failures, "\n".join(failures)


def test_question4_not_refused_and_seven_companies(engine: Any, golden_index: dict[int, Any]) -> None:
    """captain blocker 的首个案例：题 4 必须作答（不得拒答）并覆盖 7 家企业、不得出现赵马克。

    实测背景：权威产物 ``answer_eval_t6.json`` 出现 ``answered=13``、``unknown_ids=[4]``，
    题 4 的 ``unknown_reason = subject_gate:org_leaked`` → 属**误拒答**（用户问企业清单却被告知「不清楚」）。
    """
    item = golden_index[4]
    answer = _ask(engine, item.question, session="t8-q4-blocker")
    checks = assertions.check_no_false_refusal([
        {"id": 4, "is_unknown": bool(getattr(answer, "is_unknown", False)),
         "unknown_reason": getattr(answer, "unknown_reason", None)}
    ], first_case=4)
    checks += assertions.check_leakage(str(answer.text),
                                       required=item.required_substrings,
                                       forbidden=item.forbidden_substrings, label="题 4")
    gate = getattr(answer, "subject_gate", None)
    if gate is not None:
        checks += assertions.check_gate_expectation(gate, 4)
    write_report("user_question4_blocker", "题 4 误拒答专项（首案例）", [
        {"id": 4, "question": item.question, "text": str(answer.text),
         "is_unknown": bool(getattr(answer, "is_unknown", False)),
         "unknown_reason": getattr(answer, "unknown_reason", None),
         "gate": (gate.to_dict() if hasattr(gate, "to_dict") else None),
         "checks": [c.to_dict() for c in checks]}
    ])
    bad = [c for c in checks if not c.ok]
    assert not bad, ("题 4 未通过（captain 裁定的首个误拒答案例）：\n"
                     + "\n".join(c.render() for c in bad))


def test_subject_gate_fail_open_for_text_evidence(engine: Any, golden_index: dict[int, Any]) -> None:
    """N-6：题 34 证据在正文段（物理 152，tables=0）→ 闸门必须 fail-open 且不得拒答。

    ⚠️ 当前**预期为红**：``Answer.subject_gate`` 字段缺失（冻结契约 §3.19 v1.2，t15 修复中），
    导致「闸门 fail-open / counted=False」无法审计。**注意**：本表现在**能**正确作答、未被误拒答，
    因此该红只反映「字段缺失、不可审计」，不等于 N-6 行为错误。
    """
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for question_id in (34, 793):
        item = golden_index[question_id]
        answer = _ask(engine, item.question, session=f"t8-gate-{question_id}")
        gate = getattr(answer, "subject_gate", None)
        rows.append({"id": question_id, "is_unknown": bool(getattr(answer, "is_unknown", False)),
                     "gate": (gate.to_dict() if hasattr(gate, "to_dict") else
                              (dict(gate) if isinstance(gate, dict) else str(gate))),
                     "text": str(answer.text)})
        if bool(getattr(answer, "is_unknown", False)):
            failures.append(f"题 {question_id} 被误判为不可答（N-6 违反）：{answer.text[:60]!r}")
        if gate is None:
            failures.append(f"题 {question_id} 未返回 subject_gate 字段（闸门结果无法审计）"
                            f"【性质：产品缺陷 —— 缺冻结契约字段 Answer.subject_gate，t15 修复中】")
            continue
        checks = assertions.check_subject_gate_fail_open(gate, label=f"题 {question_id}")
        failures += [c.render() for c in checks if not c.ok]
    write_report("user_subject_gate", "闸门 fail-open 取证", rows)
    assert not failures, "\n".join(failures)


def test_pdf2_bank_loan_not_mixed_with_ipo_proceeds(engine: Any, discovered_pdfs: list[Any]) -> None:
    """红线⑨：问 PDF2 的「补流借款」不得答成题 207 的 15,000 万元，也不得引 PDF1。"""
    pdf2 = [p.name for p in discovered_pdfs if p.stem.endswith("2")]
    if not pdf2:
        pytest.skip("PDF2 缺席：跨语料对照自动 pending")
    answer = _ask(engine, "武汉力源信息技术股份有限公司向汉口银行借款多少用于补充流动资金？",
                  session="t8-bank-loan", file_names=pdf2)
    text = str(answer.text)
    assert "15,000" not in text, f"把 PDF1 的募投金额答成了 PDF2 的借款：{text[:80]!r}"
    cited = [str(getattr(c, "file_name", "")) for c in (getattr(answer, "citations", []) or [])]
    assert all("招股说明书1" not in name for name in cited), f"跨语料引用 PDF1：{cited}"
    write_report("user_bank_loan", "PDF2 银行借款对照", [
        {"question": "汉口银行借款多少用于补充流动资金", "text": text,
         "is_unknown": bool(getattr(answer, "is_unknown", False)), "citations": cited,
         "note": "设计裁定：该问句可答 = 300 万美元（PDF2 物理 340）；本用例只做「不得串语料」的硬断言"}
    ])
