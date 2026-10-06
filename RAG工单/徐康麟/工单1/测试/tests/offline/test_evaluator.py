"""离线测试：评估器（``app/core/evaluator.py``）。

测试目标（工单 9.1 / 第 8 节）：
1. ``check_answer`` 判分逻辑：完全正确、答案包含参考答案、参考答案包含答案、
   数字全部命中、数字部分命中（应判错并说明缺哪些数字）、关键实体命中、比例命中；
2. ``evaluate_answer`` 能生成完整的 ``EvalRecord``（含引用与耗时）；
3. ``summarize`` 的准确率 / "不清楚"回复正确率 / 引用正确率计算正确；
4. ``write_comparison_csv`` / ``write_report`` / ``write_json`` 能真实产出报告文件
   （写入 ``tmp_path``，不污染 ``优化/评估结果/eval_results``）；
5. ``ragas_available`` 只是能力探测，不得因缺 RAGAS 而报错。
"""

from __future__ import annotations

import csv
import json

import pytest

from app.core.evaluator import Evaluator
from app.models.schemas import Answer, Citation, EvalRecord, GoldenQA

GOLDEN_CAPITAL = "注册资本为5,520万元。"
GOLDEN_RATIO = "报告期内，公司来自军用领域的收入占主营业务收入的比重分别为82.10%、97.31%、94.84%和94.34%。"
GOLDEN_AWARD = "某大型研究所牵头承担的“某情报、指挥、控制与通信网络一体化工程”荣获国家科技进步一等奖。"


@pytest.fixture
def evaluator(tmp_path) -> Evaluator:
    """评估器：报告输出目录改到临时目录，避免污染 优化/评估结果/eval_results。"""
    instance = Evaluator()
    instance.results_dir = tmp_path / "eval_results"
    instance.results_dir.mkdir(parents=True, exist_ok=True)
    return instance


# ==========================================================================
# 1. 判分逻辑
# ==========================================================================
def test_check_answer_exact_and_substring(evaluator) -> None:
    """完全一致、答案含参考答案、参考答案含答案，三种情况都应判对。"""
    ok, note = evaluator.check_answer(GOLDEN_CAPITAL, GOLDEN_CAPITAL)
    assert ok is True, f"答案与参考答案完全一致时应判对，判分说明：{note}"

    ok, note = evaluator.check_answer(f"根据招股意向书，{GOLDEN_CAPITAL}", GOLDEN_CAPITAL)
    assert ok is True, f"答案包含参考答案时应判对，判分说明：{note}"

    ok, note = evaluator.check_answer("5,520万元", GOLDEN_CAPITAL)
    assert ok is True, f"答案为参考答案子串时应判对，判分说明：{note}"


def test_check_answer_wrong_answer(evaluator) -> None:
    """数字错误的答案必须判错，并在说明中给出缺失数字。"""
    ok, note = evaluator.check_answer("注册资本为1,000万元。", GOLDEN_CAPITAL)
    assert ok is False, f"注册资本答错却判为正确，判分说明：{note}"
    assert "未命中" in note, f"判错说明应包含'未命中'，实际为：{note!r}"


def test_check_answer_missing_all_numbers_is_wrong(evaluator) -> None:
    """一个数字都没答对时必须判错，并列出缺少的数值。"""
    ok, note = evaluator.check_answer("报告期内，公司收入占主营业务收入的比重较高。", GOLDEN_RATIO)
    assert ok is False, f"完全没有给出比重的答案却判为正确，判分说明：{note}"
    # 说明文案为“未命中：缺少数值 [...]”（数值而非数字，因为金额与百分比统一按数值比对）
    assert "缺少数值" in note, f"判错说明应列出缺少的数值，实际为：{note!r}"


@pytest.mark.xfail(
    reason=(
        "判分口径偏宽的已知风险：check_answer 的第 5 条模糊相似度规则以"
        "『数字集合存在交集』作为启用条件（app/core/evaluator.py），"
        "因此只答出 2/4 个比重、文字高度相似时会被判为正确，使准确率虚高"
    ),
    strict=False,
)
def test_check_answer_partial_number_match_should_be_wrong(evaluator) -> None:
    """只命中一半数字（2/4 个比重）应判为不完整，而不是正确。"""
    partial = "报告期内，公司来自军用领域的收入占主营业务收入的比重分别为82.10%和97.31%。"
    ok, note = evaluator.check_answer(partial, GOLDEN_RATIO)
    assert ok is False, f"只答出 2/4 个比重却判为正确，判分说明：{note}"


def test_check_answer_all_numbers_hit(evaluator) -> None:
    """表述不同但数字全部命中，应判为正确。"""
    rewritten = "四个报告期的比重依次是 94.34%、94.84%、97.31%、82.10%。"
    ok, note = evaluator.check_answer(rewritten, GOLDEN_RATIO)
    assert ok is True, f"数字全部命中时应判对，判分说明：{note}"
    # 命中依据可能是“比例全部命中”（百分比路径）或“数值/金额全部命中”
    assert any(token in note for token in ("比例", "数值", "数字", "金额")), (
        f"判分说明应说明命中依据，实际为：{note!r}"
    )


def test_check_answer_entity_match(evaluator) -> None:
    """关键实体（引号内的工程名称）命中即判对，允许其余表述不同。"""
    answer = "该工程是“某情报、指挥、控制与通信网络一体化工程”，2014 年获奖。"
    ok, note = evaluator.check_answer(answer, GOLDEN_AWARD)
    assert ok is True, f"关键实体一致时应判对，判分说明：{note}"


def test_check_answer_percent_match(evaluator) -> None:
    """只问比例的问题允许仅凭比例集合命中。"""
    golden = "军用领域收入占比为 82.10%。"
    ok, note = evaluator.check_answer("占比 82.10%（其余为其他业务）", golden)
    assert ok is True, f"比例命中时应判对，判分说明：{note}"


@pytest.mark.xfail(
    reason=(
        "已知缺陷：数值按字符串集合比较（app/core/evaluator.py:177-180），"
        "『5,520.00 万元』与标准答案『5,520 万元』数值等价却判为缺少数字 5520，"
        "会让准确率被低估"
    ),
    strict=False,
)
def test_check_answer_equivalent_amount_with_decimals(evaluator) -> None:
    """数值等价的不同写法（5,520.00 与 5,520）必须判为一致。"""
    ok, note = evaluator.check_answer("注册资本为 5,520.00 万元。", "注册资本为5,520万元。")
    assert ok is True, f"数值等价的答案被判错，判分说明：{note}"


def test_check_answer_empty_inputs(evaluator) -> None:
    """答案或参考答案为空时必须判错，不得抛异常。"""
    ok, note = evaluator.check_answer("", GOLDEN_CAPITAL)
    assert ok is False and "为空" in note, f"空答案应判错并说明原因，实际：{ok}, {note!r}"
    ok, note = evaluator.check_answer("注册资本为5,520万元。", "")
    assert ok is False, f"空参考答案应判错，实际：{ok}, {note!r}"


def test_check_answer_is_self_consistent_on_golden_file(evaluator, golden_qa) -> None:
    """用标准答案自身做自检：参考答案必须能被判为正确（防止判分逻辑整体失效）。"""
    for item in golden_qa:
        ok, note = evaluator.check_answer(item.answer, item.answer)
        assert ok is True, f"题号 {item.id} 的参考答案自检失败，判分说明：{note}"


# ==========================================================================
# 2. 单题评估记录
# ==========================================================================
def _make_answer(text: str, unknown: bool = False, pages: tuple[int, ...] = (22,)) -> Answer:
    """构造一个用于评估的 Answer。"""
    citations = [Citation(page=page, chunk_id=f"c{page:06d}", snippet="原文片段") for page in pages]
    return Answer(
        answer=text,
        citations=[] if unknown else citations,
        is_unknown=unknown,
        first_token_ms=12.5,
        total_ms=30.0,
    )


def test_evaluate_answer_builds_record(evaluator) -> None:
    """单题评估必须产出字段齐全的 EvalRecord。"""
    golden = GoldenQA(id=543, question="注册资本是多少？", answer=GOLDEN_CAPITAL, evidence_pages=[22])
    record = evaluator.evaluate_answer(golden, _make_answer(GOLDEN_CAPITAL), mode="extractive")

    assert isinstance(record, EvalRecord), "评估结果必须是 EvalRecord 模型"
    assert record.question_id == 543 and record.mode == "extractive", "评估记录的题号或模式不正确"
    assert record.is_correct is True, f"参考答案与回答一致却判错：{record.answer!r}"
    assert record.citation_pages == [22], f"引用页码记录错误：{record.citation_pages}"
    assert record.citation_valid is True, "引用带 chunk_id 时应视为有效引用"
    assert record.first_token_ms == 12.5, "首字耗时未写入评估记录"


def test_evaluate_answer_respects_should_be_unknown(evaluator) -> None:
    """标注为"应拒答"的题目：答了反而判错，拒答才判对。"""
    golden = GoldenQA(id=1, question="今天天气怎么样？", answer="", should_be_unknown=True)

    wrong = evaluator.evaluate_answer(golden, _make_answer("今天晴，25 度。"), mode="extractive")
    assert wrong.is_correct is False, "该题应拒答，模型给出答案时必须判错"

    right = evaluator.evaluate_answer(golden, _make_answer("不清楚", unknown=True), mode="extractive")
    assert right.is_correct is True, "该题应拒答，模型回复不清楚时必须判对"


def test_evaluate_answer_detects_invalid_citation(evaluator) -> None:
    """引用缺少 chunk_id（疑似幻觉）时，citation_valid 必须为 False。"""
    golden = GoldenQA(id=543, question="注册资本是多少？", answer=GOLDEN_CAPITAL, evidence_pages=[22])
    answer = Answer(answer=GOLDEN_CAPITAL, citations=[Citation(page=999, chunk_id="", snippet="")])
    record = evaluator.evaluate_answer(golden, answer, mode="extractive")
    assert record.citation_valid is False, "无 chunk_id 的引用应判为无效引用"


def test_evaluate_answer_counts_valid_citations(evaluator) -> None:
    """引用有效性必须逐条统计：5 条里只有 1 条有效，不能算整题有效。"""
    golden = GoldenQA(id=543, question="注册资本是多少？", answer=GOLDEN_CAPITAL, evidence_pages=[22])
    answer = Answer(
        answer=GOLDEN_CAPITAL,
        citations=[
            Citation(page=22, chunk_id="c000022", snippet="注册资本 5,520 万元"),
            Citation(page=999, chunk_id="", snippet=""),
        ],
    )
    record = evaluator.evaluate_answer(golden, answer, mode="extractive")

    assert record.citation_pages == [22, 999], f"引用页码应逐条记录，实际 {record.citation_pages}"
    assert record.citation_valid_count == 1, (
        f"有效引用条数应为 1，实际 {record.citation_valid_count}"
    )
    assert record.citation_valid is False, "存在无来源的引用时，整题引用不能判为全部有效"


# ==========================================================================
# 3. 汇总指标
# ==========================================================================
def test_summarize_computes_metrics() -> None:
    """汇总指标：准确率、拒答正确率、引用正确率、首字延迟统计。"""
    evaluator = Evaluator()
    records = [
        EvalRecord(question_id=1, question="q1", mode="extractive", answer="a", golden="g", is_correct=True,
                   citation_pages=[1], citation_valid=True, citation_valid_count=1,
                   first_token_ms=10.0, total_ms=20.0),
        EvalRecord(question_id=2, question="q2", mode="extractive", answer="b", golden="g", is_correct=False,
                   citation_pages=[2], citation_valid=False, citation_valid_count=0,
                   first_token_ms=30.0, total_ms=40.0),
        EvalRecord(question_id=3, question="q3", mode="extractive", answer="不清楚", golden="", is_correct=True,
                   is_unknown=True, should_be_unknown=True, citation_pages=[], citation_valid=False,
                   citation_valid_count=0, first_token_ms=20.0, total_ms=25.0),
    ]
    summaries = evaluator.summarize(records)
    assert set(summaries) == {"extractive"}, f"应按模式分组汇总，实际分组：{set(summaries)}"

    summary = summaries["extractive"]
    assert summary.count == 3, f"题目数应为 3，实际 {summary.count}"
    assert summary.accuracy == pytest.approx(2 / 3, abs=1e-4), f"准确率应为 66.67%，实际 {summary.accuracy}"
    assert summary.citation_total == 2, f"引用总条数应为 2，实际 {summary.citation_total}"
    assert summary.citation_valid == 1, f"有效引用条数应为 1，实际 {summary.citation_valid}"
    assert summary.citation_accuracy == pytest.approx(0.5, abs=1e-4), (
        f"引用正确率应为 1/2（按引用条数计），实际 {summary.citation_accuracy}"
    )
    assert summary.unknown_accuracy == pytest.approx(1.0, abs=1e-4), (
        "三题的拒答判定都与预期一致，拒答正确率应为 100%"
    )
    assert summary.first_token_max_ms == 30.0, f"最大首字延迟应为 30ms，实际 {summary.first_token_max_ms}"
    assert summary.first_token_avg_ms == pytest.approx(20.0, abs=1e-4), "平均首字延迟计算错误"


def test_summarize_empty_records() -> None:
    """空记录集不得抛异常（除零保护）。"""
    assert Evaluator().summarize([]) == {}, "空记录应返回空汇总"


# ==========================================================================
# 4. 报告输出
# ==========================================================================
def test_write_comparison_csv(evaluator, tmp_path) -> None:
    """RAG vs 纯 LLM 对比 CSV 必须能生成且包含关键列。"""
    records = [
        EvalRecord(question_id=543, question="注册资本是多少？", mode="extractive",
                   answer=GOLDEN_CAPITAL, golden=GOLDEN_CAPITAL, is_correct=True, citation_pages=[22],
                   citation_valid=True, first_token_ms=12.0, total_ms=20.0),
        EvalRecord(question_id=543, question="注册资本是多少？", mode="llm",
                   answer="不知道", golden=GOLDEN_CAPITAL, is_correct=False, first_token_ms=800.0, total_ms=900.0),
    ]
    target = evaluator.write_comparison_csv(records)
    assert target.exists() and target.stat().st_size > 0, f"对比 CSV 未生成：{target}"
    assert target.parent == evaluator.results_dir, "报告必须写入配置的结果目录（测试中已重定向到 tmp_path）"

    with open(target, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, f"同一题号应合并成一行，实际 {len(rows)} 行"
    assert rows[0]["question_id"] == "543", "CSV 中的题号不正确"
    assert rows[0]["rag_answer"] == GOLDEN_CAPITAL, "CSV 中缺少 RAG 回答"
    assert rows[0]["llm_answer"] == "不知道", "CSV 中缺少纯 LLM 回答"
    assert rows[0]["rag_correct"] == "1", "CSV 中 RAG 判分结果错误"


def test_write_report_markdown(evaluator) -> None:
    """Markdown 评估报告必须包含确定性指标表、RAGAS 小节与逐题结果。"""
    records = [
        EvalRecord(question_id=543, question="注册资本是多少？", mode="extractive",
                   answer=GOLDEN_CAPITAL, golden=GOLDEN_CAPITAL, is_correct=True, citation_pages=[22],
                   citation_valid=True, first_token_ms=12.0, total_ms=20.0),
    ]
    target = evaluator.write_report(records)
    content = target.read_text(encoding="utf-8")

    assert target.exists(), f"评估报告未生成：{target}"
    assert "# RAG vs 纯 LLM 评估报告" in content, "报告缺少标题"
    assert "## 1. 确定性指标" in content, "报告缺少确定性指标小节"
    assert "## 2. RAGAS 指标" in content, "报告缺少 RAGAS 小节"
    assert "未运行" in content, "未运行 RAGAS 时报告必须显式标注『未运行』，不得伪造数值"
    assert "543" in content, "报告缺少逐题结果"


def test_write_report_with_ragas_result(evaluator) -> None:
    """提供 RAGAS 结果时，四项指标应写入报告。"""
    records = [
        EvalRecord(question_id=543, question="注册资本是多少？", mode="extractive",
                   answer=GOLDEN_CAPITAL, golden=GOLDEN_CAPITAL, is_correct=True,
                   citation_pages=[22], citation_valid=True),
    ]
    target = evaluator.write_report(records, ragas_result={"faithfulness": 0.9, "context_recall": 0.8})
    content = target.read_text(encoding="utf-8")
    assert "faithfulness" in content and "0.9" in content, "RAGAS 指标未写入报告"
    assert "context_recall" in content and "0.8" in content, "RAGAS 指标未写入报告"


def test_write_json_records(evaluator) -> None:
    """评估明细 JSON 必须可解析，且同时包含汇总与逐题记录。"""
    records = [
        EvalRecord(question_id=543, question="注册资本是多少？", mode="extractive",
                   answer=GOLDEN_CAPITAL, golden=GOLDEN_CAPITAL, is_correct=True, citation_pages=[22],
                   citation_valid=True),
    ]
    target = evaluator.write_json(records)
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert "summaries" in payload and "records" in payload, "评估明细缺少 summaries 或 records"
    assert len(payload["records"]) == 1, "评估明细中的记录数不正确"
    assert payload["records"][0]["question_id"] == 543, "评估明细中的题号不正确"
    assert "accuracy" in payload["summaries"]["extractive"], "汇总中缺少准确率字段"


def test_ragas_available_is_boolean(evaluator) -> None:
    """RAGAS 可用性探测必须返回布尔值，缺依赖时不得抛异常。"""
    available = evaluator.ragas_available()
    assert isinstance(available, bool), f"ragas_available() 应返回布尔值，实际 {type(available)}"


def test_run_ragas_returns_none_on_empty_samples(evaluator) -> None:
    """RAGAS 样本为空时应返回 None（不伪造指标）。"""
    assert evaluator.run_ragas([]) is None, "空样本时必须返回 None，不得编造 RAGAS 指标"
