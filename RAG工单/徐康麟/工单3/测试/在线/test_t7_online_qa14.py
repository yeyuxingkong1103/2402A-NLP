# -*- coding: utf-8 -*-
"""在线级：14 题端到端问答（准确率 / 误拒答 / 逐题首字 / 引用可回溯）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定口径：
    * 准确率 = ``evaluator_bridge.check_answer``（工单1 Evaluator 口径，FUZZY_THRESHOLD=0.62），
      门槛 **≥13/14 = 92.9% ≥ 90%**（验收 3）；
    * 首字 = ``Answer.first_token_ms`` **逐题** ≤ 3000 ms（验收 4，取最大值判定，冷启动不豁免）；
    * 引用 = 四点核验（文件名真实 / 1-based 物理页在范围 / 该页含支撑原文 / 非 0-based 错页），
      ``citation_accuracy`` 必须 = 1.0（验收 5）；
    * 14 题误拒答 = 0（验收 6）。

说明：``required_substrings`` 要素齐备检查只作**诊断留痕**（不参与准确率门槛），
避免用自造判据覆盖 captain 已裁定的 ``check_answer`` 口径。
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from common import assertions, judge_local, paths
from common.reports import write_report

pytestmark = [pytest.mark.online, pytest.mark.slow]


def test_frozen_judge_module_available() -> None:
    """冻结判分模块 ``app/core/evaluator_bridge.py``（设计 §3.23）必须存在。

    它是验收 3「14 题准确率 ≥ 90%」的**唯一合规判分口径**；缺失时只能用
    「工单1 权威 Evaluator（只读）」兜底 —— 那属于降级，必须在报告里注明（不得冒充产品口径）。
    """
    available, detail = judge_local.judge_source_available()
    assert available, f"判分口径被降级：{detail}"


# ``chunk_lookup`` 已上移到 ``测试/在线/conftest.py``（qa14 与英文覆盖两个模块共用）


@pytest.fixture(scope="session")
def qa_results(engine: Any, runnable_items: list[Any], page_lookup: Any,
               page_counts: dict[str, int], chunk_lookup: dict[str, Any]) -> list[dict[str, Any]]:
    """跑一遍 14 题并缓存逐题结果（含判定、引用核验、要素诊断），同时写留痕。

    ★ 判分输入取 **答案正文**（``citation.answer_body``）：§2.1 判的是答案内容，引用由 §2.4 单独判；
    直接把渲染后的整段（含末尾 ``引用：[文件名: 页码]``）丢给判分器会额外引入页码等数字，
    实测会把「答案就是标准名」的题 95、以及「仅前缀不同」的题 793 误判为错（公平性缺陷，已纠正）。
    """
    paths.ensure_dev_on_path()
    from app.core.citation import answer_body  # noqa: PLC0415

    files = list(page_counts)
    rows: list[dict[str, Any]] = []
    for item in runnable_items:
        started = time.perf_counter()
        answer = engine.ask(item.question, session_id=f"t8-q{item.id}", stream=False)
        wall_ms = round((time.perf_counter() - started) * 1000, 2)
        body = answer_body(str(answer.text))
        correct, reason, source = assertions.judge_answer(body, item.answer)
        correct_full, reason_full, _ = assertions.judge_answer(str(answer.text), item.answer)

        citations = list(getattr(answer, "citations", []) or [])
        cite_rows: list[dict[str, Any]] = []
        for cite in citations:
            check = assertions.check_citation_traceable(
                cite, page_lookup=page_lookup, page_counts=page_counts,
                discovered_files=files, chunk_lookup=chunk_lookup)
            cite_rows.append({"file_name": getattr(cite, "file_name", ""),
                              "page": getattr(cite, "page", None),
                              "chunk_id": getattr(cite, "chunk_id", ""),
                              "quote": str(getattr(cite, "quote", "") or "")[:60],
                              "ok": check.ok, "detail": check.detail})

        element_checks = assertions.check_leakage(
            str(answer.text), required=item.required_substrings,
            forbidden=item.forbidden_substrings, label=f"题 {item.id}")
        gate = getattr(answer, "subject_gate", None)
        rows.append({
            "id": item.id, "corpus": item.corpus, "question": item.question,
            "answer_text": str(answer.text),
            "answer_body": body,
            "is_unknown": bool(getattr(answer, "is_unknown", False)),
            "unknown_reason": getattr(answer, "unknown_reason", None),
            "first_token_ms": float(getattr(answer, "first_token_ms", 0.0) or 0.0),
            "total_ms": float(getattr(answer, "total_ms", 0.0) or 0.0),
            "wall_ms": wall_ms,
            "backend": getattr(answer, "backend", ""), "model": getattr(answer, "model", ""),
            "language": getattr(answer, "language", ""),
            "correct": bool(correct), "judge_reason": reason, "judge_source": source,
            "correct_full_text": bool(correct_full), "judge_reason_full_text": reason_full,
            "golden_answer": item.answer,
            "citations": cite_rows,
            "subject_gate": (gate.to_dict() if hasattr(gate, "to_dict")
                             else (dict(gate) if isinstance(gate, dict) else None)),
            "elements": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in element_checks],
            "elements_ok": all(c.ok for c in element_checks),
        })

    write_report("online_qa14", "14 题端到端问答结果", rows,
                 extra={
                     "judge": "§2.1 五步口径（产品 bridge → 工单1 权威 Evaluator（只读）→ 本地副本）",
                     "judge_input": "答案正文（citation.answer_body）；另存含引用行的判定结果作对照",
                     "budget_ms": 3000,
                     "correct_body": sum(1 for row in rows if row["correct"]),
                     "correct_full_text": sum(1 for row in rows if row["correct_full_text"]),
                 })
    return rows


def test_accuracy_gate(qa_results: list[dict[str, Any]]) -> None:
    """验收 3：语义正确 ≥ 13/14 且准确率 ≥ 0.90。"""
    correct = sum(1 for row in qa_results if row["correct"])
    wrong = [(row["id"], row["judge_reason"]) for row in qa_results if not row["correct"]]
    checks = assertions.accuracy_gate(correct, len(qa_results))
    assert all(check.ok for check in checks), (
        f"准确率未达标：{correct}/{len(qa_results)}；判错题 {wrong}；"
        f"逐题见 测试/留痕/online_qa14.json")


def test_no_false_refusal(qa_results: list[dict[str, Any]]) -> None:
    """验收 6 反向 + 误拒答 blocker：14 题 ``unknown_ids`` 必须为空，首案例 = 题 4。

    captain 实测（``answer_eval_t6.json`` 17:05:20）：``answered=13``、``unknown_ids=[4]``，
    题 4 被主体闸门以 ``subject_gate:org_leaked`` 误拒答 → 这是**误拒答**，不是「不清楚」。
    """
    checks = assertions.check_no_false_refusal(qa_results)
    bad = [check for check in checks if not check.ok]
    assert not bad, ("\n".join(c.render() for c in bad)
                     + "\n（题 4 = 首个案例；详见 测试/留痕/online_qa14.json 的 subject_gate 字段）")


def test_question4_subject_gate_counted_and_not_refused(qa_results: list[dict[str, Any]]) -> None:
    """题 4 闸门必须真正生效且不误拒答：ok=True、expected=organization、counted=True、allowed ⊇ 7 家。

    ⚠️ 当前**预期为红**：``Answer.subject_gate`` 字段缺失（设计 §3.19 v1.2 冻结、§10.2 依赖），
    属**产品侧实现偏离冻结契约**（captain 已派 t15 补齐）。字段到位后本用例应自动转绿。
    """
    row = next(r for r in qa_results if r["id"] == 4)
    gate = row.get("subject_gate")
    assert gate is not None, ("题 4 未返回 subject_gate 字段（闸门结果无法审计）"
                              "【性质：产品缺陷 —— 冻结契约 §3.19 v1.2 要求 Answer.subject_gate，t15 修复中】")
    checks = assertions.check_gate_expectation(gate, 4)
    bad = [check for check in checks if not check.ok]
    assert not bad, ("题 4 闸门期望不满足：\n" + "\n".join(check.render() for check in bad)
                     + f"\n实际 gate={gate}")


def test_question3_subject_gate_counted_and_not_refused(qa_results: list[dict[str, Any]]) -> None:
    """题 3 闸门必须真正生效且不误拒答：ok=True、expected=person、counted=True、allowed ∋ 赵马克。

    ⚠️ 当前**预期为红**：同题 4，根因是 ``Answer.subject_gate`` 字段缺失（产品侧，t15 修复中）。
    """
    row = next(r for r in qa_results if r["id"] == 3)
    gate = row.get("subject_gate")
    assert gate is not None, ("题 3 未返回 subject_gate 字段（闸门结果无法审计）"
                              "【性质：产品缺陷 —— 冻结契约 §3.19 v1.2 要求 Answer.subject_gate，t15 修复中】")
    checks = assertions.check_gate_expectation(gate, 3)
    bad = [check for check in checks if not check.ok]
    assert not bad, ("题 3 闸门期望不满足：\n" + "\n".join(check.render() for check in bad)
                     + f"\n实际 gate={gate}")


def test_first_token_per_question(qa_results: list[dict[str, Any]]) -> None:
    """验收 4：**每题**首字 ≤ 3000 ms（逐题判定，取最大值；冷启动不豁免）。"""
    latencies = {f"题 {row['id']}": row["first_token_ms"] for row in qa_results}
    checks = assertions.check_first_token(latencies)
    bad = [check for check in checks if not check.ok]
    assert not bad, ("首字超预算：\n" + "\n".join(c.render() for c in bad)
                     + "\n" + assertions.first_token_hint())


def test_citations_present_and_traceable(qa_results: list[dict[str, Any]]) -> None:
    """验收 5：每题都有引用，且引用正确率 = 1.0（逐条可回溯到真实物理页）。"""
    missing = [row["id"] for row in qa_results if not row["citations"]]
    assert not missing, f"以下题没有引用：{missing}"
    checks = [assertions.Check(f"引用 {row['file_name']}:{row['page']}（题 {r['id']}）",
                               row["ok"], row["detail"])
              for r in qa_results for row in r["citations"]]
    accuracy = assertions.citation_accuracy(checks)
    failures = [c.render() for c in checks if not c.ok]
    assert accuracy == 1.0, (f"引用正确率 {accuracy:.4f} < 1.0；失败引用：\n" + "\n".join(failures))


def test_question_207_rules(qa_results: list[dict[str, Any]]) -> None:
    """题 207 专项：含 15,000；不得出现 PDF2 物理 340 的银行借款内容；只引 PDF1。"""
    row = next(r for r in qa_results if r["id"] == 207)
    text = row["answer_text"]
    assert "15,000" in text, f"题 207 未给出 15,000：{text[:80]!r}"
    for token in ("300 万美元", "汉口银行", "民生银行"):
        assert token not in text, f"题 207 混入了 PDF2 银行借款内容：{token!r}"
    bad = [c for c in row["citations"] if "招股说明书2" in str(c["file_name"])]
    assert not bad, f"题 207 引用了 PDF2（红线⑨禁止）：{bad}"


def test_pdf2_questions_cite_pdf2(qa_results: list[dict[str, Any]]) -> None:
    """id 1~4 的引用必须落在 PDF2（力源信息）上。"""
    problems: list[str] = []
    for row in qa_results:
        if row["id"] not in (1, 2, 3, 4):
            continue
        if not any("招股说明书2" in str(c["file_name"]) for c in row["citations"]):
            problems.append(f"题 {row['id']} 的引用未落在 PDF2：{row['citations']}")
    assert not problems, "\n".join(problems)


def test_enumerated_answers_are_complete(qa_results: list[dict[str, Any]]) -> None:
    """**枚举型问题必须给全枚举**（captain 裁定 2026-10-04 的产品侧兜底）。

    为什么要这条：§2.1 判分器第①步偏松（只答 1 个比重也能被判对），captain 裁定**不修判分器**
    （它是与工单2 基线同口径的尺子），因此「答案完整」必须由产品侧自检 + 本硬断言兜底：
    题 33 要先给全 4 个比重、题 260 给全 4 个金额、题 2 给全 5 个项目、题 4 给全 7 家企业……。
    非枚举型题目仍只作**诊断留痕**，不参与门槛（避免用自造判据覆盖 §2.1 口径）。
    """
    enumerated = set(assertions.ENUMERATED_QUESTIONS)
    problems: list[str] = []
    for row in qa_results:
        if row["id"] not in enumerated:
            continue
        missing = [e["name"] for e in row["elements"] if not e["ok"]]
        if missing:
            problems.append(f"题 {row['id']} 枚举不完整：缺 {missing}")
    assert not problems, ("枚举完整性未达标（产品应给全枚举，不得依赖判分器漏放）：\n"
                          + "\n".join(problems))


def test_element_completeness_is_recorded(qa_results: list[dict[str, Any]]) -> None:
    """非枚举题的要素齐备**只作诊断留痕**；后端/模型字段必须齐备。"""
    weak = [(row["id"], [e["name"] for e in row["elements"] if not e["ok"]])
            for row in qa_results if not row["elements_ok"]]
    if weak:
        print("要素诊断（非门槛，供 T9/T11 参考）：", weak)
    assert all("backend" in row for row in qa_results)
