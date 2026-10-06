"""T3 离线测试 ⑦：评估层（判分口径 + 确定性指标 + 生成路径对照）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md §2 判分口径、验收 1/3/7；环境事实 §4.2）：

1. **RAGAS 未运行**：本机 ``ragas`` 不可用且断网，必须显式声明「RAGAS 未运行
   （依赖不可用且本机断网）」，**严禁伪造 RAGAS 数值**；
2. **判分口径未被放宽**：强制负例 N-1~N-4 必须判错、正例 P-1~P-3 必须判对、
   ``FUZZY_THRESHOLD`` 必须仍为 0.62（改阈值等同于篡改评估口径）；
3. **口径固有容忍**：人为插错别字的「法定代表人是程家的明」实测判 True
   （二元组 0.70 ≥ 0.62）——这是工单1 口径的**固有容忍度**，不是工单2 放宽的结果，
   此处固化为**边界用例**；引用标签 ``[页码: X]`` 的位置**不影响**判分；
4. **生成路径对照**（captain 指定的回归证据）：抽取式路径在 10 题上的判对率，
   作为「生成层是瓶颈」的可回归证据（不依赖 LLM 是否可用）；
5. 用确定性指标（准确率/检索命中率/引用正确率/首字延迟）出报告到
   ``优化/评估结果/``。
"""

from __future__ import annotations

import json
import re
import time

import pytest

from conftest import EVAL_RESULTS

#: 强制引用标签（中文 ``[页码: N]`` / 英文 ``[Page: N]``）
CITATION_LABEL_RE = re.compile(r"\[\s*(?:页码|Page)\s*[:：]?\s*\d+\s*\]")

#: 必须原样出现在产出物中的声明（环境事实 2.2 / 红线）
RAGAS_NOT_RUN_TEXT = "RAGAS 未运行（依赖不可用且本机断网）"

# ---------------------------------------------------------------- 负例（必须判错）
NEGATIVE_CASES = [
    ("N-1 多值漏答",
     "报告期内，公司来自军用领域的收入占主营业务收入的比重分别为82.10%、97.31%、94.84%和94.34%。",
     "[页码: 129] 82.10%"),
    ("N-2 单位错",
     "注册资本为5,520万元。",
     "注册资本为5,520元。"),
    ("N-2b 量级错（150 万 vs 15,000 万）",
     "公司计划使用本次发行募集资金15,000.00万元用于补充流动资金。",
     "[页码: 490] 计划使用本次发行募集资金150.00万元用于补充流动资金。"),
    ("N-3 答非所问",
     "下游行业为各类终端用户，覆盖范围广泛，主要包括军队、政府机关、能源等行业企业。",
     "公司的主要客户为军方及系统集成单位。"),
    ("N-4 只给合计",
     "报告期内，公司来自军用领域的收入分别为6,464.51万元、14,414.16万元、18,780.67万元和4,627.14万元。",
     "报告期内公司来自军用领域的收入合计为44,286.48万元。"),
]

# ---------------------------------------------------------------- 正例（必须判对）
POSITIVE_CASES = [
    ("P-1a 金额跨单位等价（1.5 亿 ↔ 15,000 万）",
     "公司计划使用本次发行募集资金15,000.00万元用于补充流动资金。",
     "[页码: 479] 1.5 亿元"),
    ("P-1b 金额等价（多前缀完整句）",
     "注册资本为5,520万元。",
     "[页码: 52] 武汉兴图新科电子股份有限公司注册资本：5,520 万元。"),
    ("P-2a 小数位等价（15,000.00 ↔ 15,000）",
     "公司计划使用本次发行募集资金15,000.00万元用于补充流动资金。",
     "[页码: 490] 15,000 万元"),
    ("P-2b 小数位等价（5,520 ↔ 5,520.00）",
     "注册资本为5,520万元。",
     "[页码: 52] 注册资本为5,520.00万元。"),
    ("P-3 完整句（参考答案为答案子串）",
     "法定代表人是程家明。",
     "[页码: 52] 武汉兴图新科电子股份有限公司法定代表人是程家明。"),
]


@pytest.fixture(scope="module")
def extractive_run(engine, evaluator, golden, chunks):
    """抽取式路径跑完 10 题（确定性，不调用 LLM），返回逐题记录与耗时。

    走**端到端编排引擎**（``QAEngine(force_extractive=True)``）而非直接调 generator：
    因为 ``Answer.citations`` 由编排层的 ``CitationManager`` 附着
    （``app/core/qa_engine.py:345``），生成层 ``generator.py`` 不产出 citations。
    直接调生成层会得到「回答文本里有 ``[页码: N]`` 但 citations 为空」的中间态，
    那不是产品路径，也不是引用正确率的判定依据。
    """
    valid_ids = {c.chunk_id for c in chunks}
    records = []
    for item in golden:
        t0 = time.perf_counter()
        answer = engine.ask(item.question)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        is_correct, note = evaluator.check_answer(answer.answer, item.answer)
        cite_total = len(answer.citations)
        cite_valid = sum(1 for c in answer.citations if c.chunk_id in valid_ids and 1 <= c.page <= 548)
        records.append({
            "id": item.id, "answer": answer.answer, "is_correct": is_correct, "note": note,
            "citation_total": cite_total, "citation_valid": cite_valid,
            "elapsed_ms": round(elapsed_ms, 2), "mode": answer.mode,
            "pages": [c.page for c in answer.citations],
        })
    return records


# ============================================================ RAGAS
def test_ragas_is_not_available_and_declared_not_run(evaluator):
    """RAGAS 必须显式声明未运行；不得伪造数值（红线）。"""
    available = evaluator.ragas_available()
    print(f"\n[RAGAS] {RAGAS_NOT_RUN_TEXT}")
    print(f"[RAGAS] evaluator.ragas_available() = {available}")
    assert available is False, (
        "本机 ragas 突然可用：若确实可用请真实运行并更新文档；若不可用不得声称已运行"
    )
    try:
        import ragas  # noqa: F401
        imported = True
    except Exception:
        imported = False
    assert imported is False, "ragas 可被导入，与环境事实 2.2 不符，请复核依赖状态"


def test_existing_artifacts_contain_no_fabricated_ragas_values(evaluator):
    """仓库既有评估产物里的 RAGAS 四项指标必须为空（未运行 = 不写数值）。"""
    path = EVAL_RESULTS / "eval_records.json"
    if not path.exists():
        pytest.skip(f"评估产物尚未生成: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    assert records, "eval_records.json 无 records"
    fields = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    fabricated = [(r.get("question_id"), f) for r in records for f in fields if r.get(f) is not None]
    assert not fabricated, f"发现被伪造/填值的 RAGAS 指标: {fabricated[:5]}"
    print(f"\n[RAGAS] eval_records.json 的 {len(records)} 条记录中四项 RAGAS 指标均为 null（未伪造）")


# ============================================================ 判分口径
def test_fuzzy_threshold_unchanged(evaluator):
    """FUZZY_THRESHOLD 必须仍为 0.62——改阈值等同篡改评估口径。"""
    from app.core import evaluator as ev

    assert ev.FUZZY_THRESHOLD == 0.62, (
        f"FUZZY_THRESHOLD 被改动为 {ev.FUZZY_THRESHOLD}，会使「优化前 50%」不再可比"
    )


@pytest.mark.parametrize("name,golden_text,answer", NEGATIVE_CASES, ids=[c[0] for c in NEGATIVE_CASES])
def test_negative_cases_must_be_wrong(evaluator, name, golden_text, answer):
    """强制负例：多值漏答/单位错/答非所问/只给合计，必须判错。"""
    ok, note = evaluator.check_answer(answer, golden_text)
    print(f"\n[负例] {name}: is_correct={ok} note={note}")
    assert ok is False, f"{name} 被判对，判分口径被放宽：{note}"


@pytest.mark.parametrize("name,golden_text,answer", POSITIVE_CASES, ids=[c[0] for c in POSITIVE_CASES])
def test_positive_cases_must_be_right(evaluator, name, golden_text, answer):
    """正例：金额跨单位/小数位等价、完整句，必须判对（防止口径被收紧到不可达）。"""
    ok, note = evaluator.check_answer(answer, golden_text)
    print(f"\n[正例] {name}: is_correct={ok} note={note}")
    assert ok is True, f"{name} 被判错，判分过严：{note}"


def test_negative_case_reports_missing_values(evaluator):
    """N-1 必须走确定性数值缺失路径（而不是别的兜底），确保判据可解释。"""
    ok, note = evaluator.check_answer(
        "[页码: 129] 82.10%",
        "报告期内，公司来自军用领域的收入占主营业务收入的比重分别为82.10%、97.31%、94.84%和94.34%。",
    )
    assert ok is False
    assert "缺少数值" in note, f"N-1 的判分说明未体现数值缺失: {note}"
    for missing in ("97.31", "94.84", "94.34"):
        assert missing in note, f"{missing} 未被列入缺失数值: {note}"


def test_boundary_inherent_tolerance_typo_still_true(evaluator):
    """**边界用例（不是「判分放宽」）**：人为插错别字仍判 True。

    「法定代表人是程家的明」实测判 True，原因是字符二元组 Jaccard = 0.70 ≥ 0.62，
    落入第 5 条模糊判据。这是工单1 判分口径的**固有容忍度**
    （该步设计用于容忍「军队视频指挥」↔「国防军队视频指挥」这类同义改写），
    并非工单2 引入的放宽；**不得据此修改 FUZZY_THRESHOLD**（改了基线不可比）。
    """
    ok, note = evaluator.check_answer("法定代表人是程家的明。", "法定代表人是程家明。")
    print(f"\n[边界] 错别字答案 is_correct={ok} note={note}")
    assert ok is True, (
        "该边界用例应判 True（口径固有容忍）。若变为 False，说明阈值被改动，需立即核查"
    )
    assert "相似度" in note, f"未走第 5 条模糊判据: {note}"


def test_citation_label_position_does_not_change_scoring(evaluator):
    """引用标签 ``[页码: X]`` 的位置不影响判分（实测对照）。"""
    golden_text = "法定代表人是程家明。"
    cases = {
        "无标签": ("法定代表人是程家明。", True),
        "标签前置": ("[页码: 52] 法定代表人是程家明。", True),
        "标签后置": ("法定代表人是程家明。[页码: 52]", True),
        "基线原样（缺「是」）": ("[页码: 52] 法定代表人：程家明", False),
        "缺「是」+标签后置": ("法定代表人：程家明。[页码: 52]", False),
    }
    results = {}
    for name, (answer, expect) in cases.items():
        ok, note = evaluator.check_answer(answer, golden_text)
        results[name] = ok
        print(f"\n[标签位置] {name}: is_correct={ok}（期望 {expect}） note={note}")
        assert ok is expect, f"{name} 判定与期望不符：{note}"
    assert results["标签前置"] == results["标签后置"] == results["无标签"], (
        "引用标签位置改变了判分结果——与实测结论不符"
    )


# ============================================================ 生成路径对照 + 确定性指标
@pytest.mark.slow
def test_extractive_failures_are_citation_label_interaction(extractive_run, evaluator, golden):
    """**文档化断言（口径交互，非能力缺口）**：抽取式路径的未通过项源于
    「强制引用标签 ``[页码: X]`` × 冻结判分口径」的交互。

    机制（T3 独立复现，与 `环境事实.md` §4.1.6 一致）：
    1. ``PUNCT_TO_STRIP`` 含 ``[`` ``]``，但**不含「页」「码」**，因此
       ``[页码: 160]`` 归一化后残留 ``页码160``，破坏第 1 条「去标点包含」判据；
    2. 第 5 条模糊判据又因参考答案的 ``1.0`` 规范化成 ``1`` 而报「缺少数值 ['1']」。

    实测：同一批答案**去掉标签后 10/10 —— 仅用于说明失败原因，非交付指标**，未通过项恰为去掉标签后转判对者。
    因此本用例断言的是「失败可被引用标签完整解释」，而**不**把它留成长期硬失败——
    后者会掩盖真正的能力回归（例如英文提问被拒答的缺陷）。

    裁定（**已裁定，非待定**）：用户已采用**方案 A**——``Answer.answer`` **保留**
    ``[页码: X]``，主指标 = **9/10（90%）**，达标；Q95 属**已知遗留**，根因是标签交互、
    **非能力缺口**。**方案 B（标签移入 citations）已否决，不得实施**。
    任何报告中**禁止**把「去标签后 10/10」当作成绩出现，只能与「非交付指标」并列说明机制。
    本用例在方案 A 下是「失败可被完整解释」的**长期支撑证据**，不是临时措施。

    **前提与边界（避免误用，勿删）**：本用例断言的是「失败可由引用标签交互**完整解释**」，
    **不是性能回归检测**。若将来系统原样即 10/10 全对，则 ``raw_failed`` 与 ``flipped``
    均为空集，两条断言会**空真通过**——这是可接受的语义退化，符合本用例声明。
    真正的能力回归检测由两条用例承担：
    ① ``test_extractive_path_meets_acceptance_threshold``（硬断言 ≥90%）；
    ② ``test_multilingual.py::test_english_questions_answered_in_english_with_real_citations``
    （英文作答契约，当前为红，t11 修复中）。
    """
    golden_by_id = {g.id: g for g in golden}
    raw_ok = sum(1 for r in extractive_run if r["is_correct"])
    stripped_ok = 0
    flipped: list[int] = []
    print("\n[引用标签交互] 逐题：原样 vs 去掉 [页码: N] 标签")
    for r in extractive_run:
        stripped = CITATION_LABEL_RE.sub("", r["answer"])
        ok, note = evaluator.check_answer(stripped, golden_by_id[r["id"]].answer)
        stripped_ok += ok
        if ok and not r["is_correct"]:
            flipped.append(r["id"])
        print(f"   Q{r['id']:>4} 原样={'✅' if r['is_correct'] else '❌'} "
              f"去标签={'✅' if ok else '❌'}  去标签判分={note[:44]}")
    raw_failed = [r["id"] for r in extractive_run if not r["is_correct"]]
    print(f"[引用标签交互] 原样 {raw_ok}/{len(extractive_run)} → 去标签 {stripped_ok}/{len(extractive_run)}")
    print(f"[引用标签交互] 原样未通过={raw_failed}；其中去掉标签后转判对={flipped}")

    assert stripped_ok == len(extractive_run), (
        f"去掉引用标签后应 10/10 全对（证明失败可由标签交互完整解释），实际 {stripped_ok}/{len(extractive_run)}"
    )
    assert set(flipped) == set(raw_failed), (
        f"未能把全部原样失败解释为标签交互：原样失败={raw_failed}，转判对={flipped}"
    )


@pytest.mark.slow
def test_extractive_path_meets_acceptance_threshold(extractive_run):
    """工单达标线（≥90%）硬断言：抽取式路径判对率必须 ≥90%。

    实测 9/10 = 90%，达标。未通过项的原因由
    ``test_extractive_failures_are_citation_label_interaction`` 文档化说明。
    """
    correct = sum(1 for r in extractive_run if r["is_correct"])
    rate = correct / len(extractive_run)
    print(f"\n[生成路径·抽取式] 逐题判分：判对 = {correct}/{len(extractive_run)} = {rate:.0%}")
    for r in extractive_run:
        print(f"   Q{r['id']:>4} {'✅' if r['is_correct'] else '❌'} mode={r['mode']} "
              f"{r['elapsed_ms']:>8.1f}ms 引用页={r['pages']} 判分={r['note'][:40]}")
    assert rate >= 0.90, f"抽取式路径准确率应 ≥90%，实际 {correct}/{len(extractive_run)} = {rate:.0%}"


@pytest.mark.slow
def test_citation_validity_of_extractive_answers(extractive_run):
    """引用正确率：每条引用须带真实 chunk_id 且页码在 1..548 内。"""
    total = sum(r["citation_total"] for r in extractive_run)
    valid = sum(r["citation_valid"] for r in extractive_run)
    print(f"\n[引用] 有效 {valid}/{total}")
    assert total > 0, "抽取式回答没有任何引用"
    assert valid == total, f"引用正确率应 100%，实际 {valid}/{total}"


@pytest.mark.slow
def test_write_deterministic_metrics_report(extractive_run, evaluator, golden, evidence_hit_chunks,
                                            retriever, query_understanding):
    """把确定性指标写入 ``优化/评估结果/``（严禁写入 RAGAS 数值）。"""
    complete = sum(1 for r in extractive_run if r["is_correct"])
    accuracy = complete / len(extractive_run)
    cite_total = sum(r["citation_total"] for r in extractive_run)
    cite_valid = sum(r["citation_valid"] for r in extractive_run)
    cite_acc = cite_valid / cite_total if cite_total else 0.0
    lat = sorted(r["elapsed_ms"] for r in extractive_run)

    # 检索命中率：**证据块是否真的进入了本次检索的最终上下文**（不是"证据在库里存在"）
    retrieval_hits = 0
    miss_ids = []
    for item in golden:
        analysis = query_understanding.analyze(item.question)
        ctx = retriever.retrieve_multi(query_understanding.search_queries(analysis), analysis)
        got = {c.chunk.chunk_id for c in ctx}
        if got & evidence_hit_chunks[item.id]:
            retrieval_hits += 1
        else:
            miss_ids.append(item.id)

    # 首字响应（工单验收项）：**从 eval_records.json 读取**，不手工誊抄，
    # 避免把「总耗时」误标成「首字」。该字段由 研发/scripts/evaluate.py 真实产出。
    eval_json = EVAL_RESULTS / "eval_records.json"
    ft_row = "| 首字响应（工单验收项） | **未运行/产物缺失** | 需先执行 `研发/scripts/evaluate.py --mode rag` |"
    if eval_json.exists():
        try:
            summary = (json.loads(eval_json.read_text(encoding="utf-8")).get("summaries") or {}).get("rag") or {}
            ft_avg = float(summary.get("first_token_avg_ms") or 0.0)
            ft_max = float(summary.get("first_token_max_ms") or 0.0)
            budget = 3000.0
            verdict = "**满足**" if ft_max <= budget else "**不满足**"
            ft_row = (f"| 首字响应（工单验收项，源 `eval_records.json` 的 `first_token_ms`） | "
                      f"均值 **{ft_avg:.1f} ms** / 最大 **{ft_max:.1f} ms** / 预算 {budget:.0f} ms → {verdict} | "
                      f"mode=rag（最终交付路径）；与下方「端到端耗时」**不是同一指标** |")
        except Exception as exc:  # pragma: no cover - 产物损坏时如实标注
            ft_row = f"| 首字响应（工单验收项） | 读取失败：{type(exc).__name__} | 见 `eval_records.json` |"

    lines = [
        "# 离线测试·确定性指标报告",
        "",
        "> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化",
        "> 阶段：测试（T3 离线七件套）",
        f"> 生成方式：`pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -v` 真实运行产出",
        "",
        "## 0. RAGAS 声明（**红线**）",
        "",
        f"**{RAGAS_NOT_RUN_TEXT}**。",
        "",
        "本机 `ragas` 未安装且无外网（环境事实 §2.2），因此 faithfulness / answer_relevancy /",
        "context_precision / context_recall 四项指标**全部为空**，未以任何方式估算或填充。",
        "以下为可复现的**确定性指标**，作为替代证据。",
        "",
        "## 1. 确定性指标",
        "",
        "| 指标 | 实测值 | 口径 |",
        "| --- | --- | --- |",
        f"| 答案准确率 | **{accuracy:.2f}（{complete}/{len(extractive_run)}）** | 工单1 `check_answer` 确定性判分，口径未放宽；抽取式路径 |",
        f"| 检索命中率（证据进最终上下文） | **{retrieval_hits}/{len(golden)}** | 证据原文运行时匹配，chunk_id 不硬编码；未命中题号 {miss_ids or '无'} |",
        f"| 引用正确率 | **{cite_acc:.2f}（{cite_valid}/{cite_total}）** | 引用须带真实 chunk_id 且页码 ∈ [1,548] |",
        ft_row,
        f"| 抽取式路径**端到端耗时**（elapsed，**非首字**） 平均 / 最大 | "
        f"**{sum(lat)/len(lat):.1f} ms / {lat[-1]:.1f} ms** | 含查询理解 + 检索 + 抽取式生成的整体墙钟时间；"
        f"**不得与「首字响应」混读** |",
        "",
        "> 说明：上表最后一行是**单次提问从进入到返回的全过程耗时**，非首字延迟。"
        "工单验收「首字响应 ≤ 3 秒」的判定依据是 `first_token_ms`（见上一行）。",
        "",
        "## 2. 逐题明细",
        "",
        "| 题号 | 判对 | 端到端耗时(ms) | 引用页 | 判分说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in extractive_run:
        lines.append(f"| {r['id']} | {'✅' if r['is_correct'] else '❌'} | {r['elapsed_ms']:.1f} | "
                     f"{r['pages']} | {r['note'][:60]} |")
    lines += [
        "",
        "## 3. 数据质量与口径声明",
        "",
        "- `golden_qa.jsonl` 的 `evidence_pages` 存在**错标**（Q531/Q543 标 [22]、实际正文在 p52；"
        "Q795/Q207 亦有多处偏差），故引用正确性**不以其为唯一真值**，只按「页码 ∈ [1,548] 且可回查 chunk」判定。",
        "- Q95 的 `evidence` 含省略号「……」、Q207 的 `evidence` 为**合成引用文本**，两者无法用"
        "整段证据匹配，测试中改用运行时替代判据并单独标注。",
        "- 样本量 10 题，准确率分辨率为 10%；「不支持回答时回复不清楚」由 T4 的在线容错用例覆盖。",
        "- `FUZZY_THRESHOLD = 0.62` **未被修改**；判分口径与工单1 一致，故「优化前 0.50 → 优化后」可比。",
        "- **引用标签交互（口径级，已裁定：采用方案 A（保留标签），主指标 9/10）**："
        "`PUNCT_TO_STRIP` 含 `[` `]` 但不含「页」「码」，"
        "故 `[页码: 160]` 归一化后残留 `页码160`，会破坏第 1 条「去标点包含」判据。"
        "实测同一批答案**去掉标签后 10/10（仅用于说明失败原因，非交付指标）**；"
        "方案 B（标签移入 `citations`）**已否决、不得实施**。"
        "本报告主指标按方案 A 计为 **9/10**；该差异**非能力缺口**。",
        "",
        "## 4. 复现命令",
        "",
        "```powershell",
        "pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -v",
        "```",
        "",
    ]
    EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    out = EVAL_RESULTS / "离线测试_确定性指标.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    assert out.exists() and out.stat().st_size > 500, f"报告写入失败: {out}"
    text = out.read_text(encoding="utf-8")
    assert RAGAS_NOT_RUN_TEXT in text, "报告缺少 RAGAS 未运行声明"
    assert "faithfulness" in text, "报告未说明 RAGAS 四项为空"
    print(f"\n[报告] 已写入 {out}（{out.stat().st_size} 字节）")
