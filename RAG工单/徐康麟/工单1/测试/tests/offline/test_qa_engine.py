"""离线测试：端到端问答引擎（``app/core/qa_engine.py``）。

测试目标（工单 9.1 / 验收标准 2、3、5）：
1. 引擎能在**无 LLM**条件下完成 10 个工单问题的完整链路（检索 -> 生成 -> 引用 -> 落库）；
2. 每题都返回非空答案，且"不清楚"判定合理；
3. 引用页码必须落在 PDF 真实页码 1..548 之内；
4. 首字返回时间 < 3000ms（工单硬指标）；
5. 空索引时统一回复"不清楚"，不得抛异常；
6. 流式接口的事件序列与最终答案契约；
7. 会话管理、检索留痕、反馈接口可用；
8. **测试不得污染真实索引库**（引擎使用 ``data/index`` 的副本运行）。

索引缺失时整个模块跳过：
``索引未就绪：请先运行 python scripts/build_index.py``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.evaluator import Evaluator

# 目录结构：<root>/测试/tests/<子目录>/xxx.py
#   parents[0]=子目录 parents[1]=tests parents[2]=测试 parents[3]=项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "研发"
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden_qa.jsonl"

# 工单第 7 节固定的 10 个题号
EXPECTED_QUESTION_IDS = {260, 95, 33, 34, 957, 793, 795, 543, 531, 207}

# --------------------------------------------------------------------------
# 已知答案偏差：使用哈希降级向量时曾有 3 题判分不通过；
# 当前索引已换成真实语义模型（bge-small-zh-v1.5），10 题全部通过，故此处为空。
# 若后续环境退化，把题号与原因登记到这里即可让对应用例标记为 xfail。
# --------------------------------------------------------------------------
KNOWN_ANSWER_MISSES: dict[int, str] = {}


def _load_golden_items():
    """读取标准答案文件（缺失时返回空列表，由固件负责跳过）。"""
    from app.models.schemas import GoldenQA

    if not GOLDEN_PATH.exists():
        return []
    return [
        GoldenQA(**json.loads(line))
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pytest_generate_tests(metafunc) -> None:
    """把工单 10 题参数化到端到端用例上。"""
    if "golden_item" not in metafunc.fixturenames:
        return
    params = []
    for item in _load_golden_items():
        marks = ()
        if item.id in KNOWN_ANSWER_MISSES:
            marks = pytest.mark.xfail(reason=f"题号 {item.id}：{KNOWN_ANSWER_MISSES[item.id]}", strict=False)
        params.append(pytest.param(item, id=f"q{item.id}", marks=marks))
    metafunc.parametrize("golden_item", params)


# ==========================================================================
# 1. 数据与引擎状态
# ==========================================================================
def test_golden_file_contains_ten_questions(golden_qa) -> None:
    """标准答案文件必须包含工单第 7 节固定的 10 道题。"""
    ids = {item.id for item in golden_qa}
    assert ids == EXPECTED_QUESTION_IDS, f"标准答案题号与工单不一致，差异：{ids ^ EXPECTED_QUESTION_IDS}"
    for item in golden_qa:
        assert item.question.strip(), f"题号 {item.id} 的问题为空"
        assert item.answer.strip(), f"题号 {item.id} 的标准答案为空"
        assert item.evidence_pages, f"题号 {item.id} 缺少证据页码"


def test_engine_stats_and_health(engine, requires_index) -> None:
    """引擎必须报告索引就绪、页数、分块数与统一的兜底文案。"""
    stats = engine.stats()
    assert stats["index_ready"] is True, "引擎未装载索引"
    assert stats["pages"] == 548, f"文档页数应为 548，实际 {stats['pages']}"
    assert stats["chunks"] > 0, "分块数为 0"
    assert stats["vector_count"] == stats["chunks"], "向量库条数与分块数不一致"
    assert stats["doc_id"], "引擎未解析出 doc_id"

    health = engine.health()
    assert health["engine"] == "ok", f"健康检查未通过：{health['engine']}"
    assert health["unknown_answer"] == "不清楚", f"兜底文案配置错误：{health['unknown_answer']}"
    assert health["first_token_budget_seconds"] == 3.0, "首字预算应为 3 秒（工单硬指标）"


# ==========================================================================
# 2. 10 题端到端契约
# ==========================================================================
def test_ask_returns_answer_within_contract(engine, golden_item, requires_index) -> None:
    """每题都必须返回非空答案、合理的不清楚标记、合法页码与可接受的首字延迟。"""
    answer = engine.ask(golden_item.question, save=False)

    assert answer.answer.strip(), f"题号 {golden_item.id} 返回了空答案"
    assert answer.is_unknown is False, (
        f"题号 {golden_item.id} 的答案存在于 PDF 中（证据页 {golden_item.evidence_pages}），"
        f"却被判定为不清楚：{answer.unknown_reason!r}"
    )
    assert answer.total_ms > 0, "未记录总耗时"
    assert answer.retrieved_count > 0, f"题号 {golden_item.id} 未召回任何片段"
    assert answer.pages, f"题号 {golden_item.id} 未记录检索命中页码"
    for page in answer.pages:
        assert 1 <= page <= 548, f"题号 {golden_item.id} 命中页码 {page} 超出 PDF 范围 1..548"

    assert answer.citations, f"题号 {golden_item.id} 的答案没有引用来源，无法追溯"
    for citation in answer.citations:
        assert 1 <= citation.page <= 548, f"引用页码 {citation.page} 超出 PDF 范围 1..548"
        assert citation.chunk_id, "引用缺少 chunk_id"
        assert citation.snippet, "引用缺少原文片段"

    assert answer.first_token_ms < 3000, (
        f"题号 {golden_item.id} 首字耗时 {answer.first_token_ms:.1f}ms 超过 3000ms 预算"
    )
    assert answer.mode in {"extractive", "fallback", "llm"}, f"未知的生成模式：{answer.mode}"


def test_ask_answer_matches_golden(engine, golden_item, requires_index) -> None:
    """逐题判分：抽取式答案与标准答案是否一致（当前环境的真实准确率）。"""
    answer = engine.ask(golden_item.question, save=False)
    is_correct, note = Evaluator().check_answer(answer.answer, golden_item.answer)
    assert is_correct, (
        f"题号 {golden_item.id} 判分不通过（{note}）；标准答案：{golden_item.answer[:80]!r}；"
        f"系统答案：{answer.answer[:120]!r}"
    )


def test_answer_accuracy_baseline(engine, golden_qa, requires_index) -> None:
    """准确率基线：10 题中至少 9 题判分通过（当前语义索引环境下实测 10/10）。

    该基线用于防止后续改动导致端到端质量退化；若某题确因环境限制无法判对，
    应先修索引/模型，而不是放宽基线。
    """
    evaluator = Evaluator()
    correct: list[int] = []
    wrong: list[tuple[int, str]] = []
    for item in golden_qa:
        answer = engine.ask(item.question, save=False)
        is_correct, note = evaluator.check_answer(answer.answer, item.answer)
        if is_correct:
            correct.append(item.id)
        else:
            wrong.append((item.id, f"{note} | 系统答案：{answer.answer[:60]!r}"))

    assert len(correct) >= 9, (
        f"端到端准确率下降：仅 {len(correct)}/{len(golden_qa)} 题判分通过（要求 >= 9）；"
        f"未通过明细：{wrong}"
    )


def test_first_token_budget_for_all_questions(engine, golden_qa, requires_index) -> None:
    """首字延迟预算（工单验收标准 5）：10 题的最大首字耗时必须 < 3000ms。"""
    timings: list[tuple[int, float]] = []
    for item in golden_qa:
        answer = engine.ask(item.question, save=False)
        timings.append((item.id, answer.first_token_ms))

    worst_id, worst_ms = max(timings, key=lambda pair: pair[1])
    average = sum(ms for _, ms in timings) / len(timings)
    assert worst_ms < 3000, (
        f"题号 {worst_id} 首字耗时 {worst_ms:.1f}ms 超过 3000ms 预算（10 题平均 {average:.1f}ms）"
    )


# ==========================================================================
# 3. 中间结果、Query 理解与流式
# ==========================================================================
def test_last_retrieved_is_recorded(engine, requires_index) -> None:
    """引擎必须留痕最近一次检索结果（供界面展示"检索片段数/页码"）。"""
    engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", save=False)
    retrieved = engine.last_retrieved()
    assert retrieved, "last_retrieved() 为空，未记录中间检索结果"
    assert all(item.chunk.page >= 1 for item in retrieved), "检索结果中存在非法页码"


def test_query_analysis_is_attached_with_intent(engine, requires_index) -> None:
    """答案必须携带 Query 理解结果（意图识别正确）。"""
    cases = {
        "武汉兴图新科电子股份有限公司注册资本是多少？": "注册资本",
        "武汉兴图新科电子股份有限公司法定代表人是谁？": "法定代表人",
    }
    for question, expected_intent in cases.items():
        answer = engine.ask(question, save=False)
        assert answer.query_analysis is not None, f"『{question}』未返回 Query 理解结果"
        assert answer.query_analysis.intent == expected_intent, (
            f"『{question}』意图识别为 {answer.query_analysis.intent!r}，期望 {expected_intent!r}"
        )


def test_stream_event_sequence(engine, requires_index) -> None:
    """流式接口必须按 first_token -> delta -> done 的顺序产出事件。"""
    conversation_id = engine.new_conversation(title="流式测试")
    events: list[tuple[str, object]] = list(
        engine.stream("武汉兴图新科电子股份有限公司注册资本是多少？", conversation_id)
    )
    names = [name for name, _ in events]
    assert names, "流式接口没有产出任何事件"
    assert "done" in names, f"流式接口未产出 done 事件：{names}"
    assert "first_token" in names, f"流式接口未产出 first_token 事件：{names}"
    assert "error" not in names, f"流式接口报错：{[payload for name, payload in events if name == 'error']}"
    assert names.index("first_token") < names.index("done"), "first_token 必须早于 done"

    final = dict(events)["done"]["answer"]  # type: ignore[index]
    assert final.answer.strip(), "流式接口的最终答案为空"
    assert final.first_token_ms >= 0, "流式接口未记录首字耗时"

    deltas = "".join(str(payload["text"]) for name, payload in events if name == "delta")  # type: ignore[index]
    assert deltas.strip(), "流式接口没有产出任何增量文本"


# ==========================================================================
# 4. 兜底、异常与落库
# ==========================================================================
def test_empty_index_returns_unknown(empty_engine) -> None:
    """索引为空时必须统一回复"不清楚"，而不是抛异常或编造答案。"""
    answer = empty_engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", save=False)
    assert answer.is_unknown is True, "无索引时应走兜底回复"
    assert answer.answer == "不清楚", f"兜底文案应为『不清楚』，实际为 {answer.answer!r}"
    assert answer.citations == [], "兜底回复不应携带引用"
    assert answer.unknown_reason, "兜底回复必须记录原因"
    assert empty_engine.health()["engine"] == "no_index", "空索引时健康状态应为 no_index"


def test_empty_question_raises_value_error(engine, requires_index) -> None:
    """空问题必须显式报错，避免把空串当成查询去检索全库。"""
    with pytest.raises(ValueError, match="问题不能为空"):
        engine.ask("   ", save=False)


def test_conversation_persistence_in_isolated_store(engine, real_message_count) -> None:
    """问答结果应写入引擎自己的库，且**不得写入真实索引库**。"""
    conversation_id = engine.new_conversation(title="落库测试")
    answer = engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", conversation_id=conversation_id)

    messages = engine.get_messages(conversation_id)
    assert len(messages) == 2, f"一次问答应落库 2 条消息，实际 {len(messages)}"
    assert messages[0].role == "user" and messages[0].content.startswith("武汉兴图新科"), "用户消息落库错误"
    assert messages[1].role == "assistant" and messages[1].content == answer.answer, "助手消息落库错误"

    # 真实索引库的 messages 行数不应因为本次测试而增加
    import sqlite3

    from app.core.config import get_settings

    db_path = get_settings().paths.sqlite_path
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15.0)
    try:
        after = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    finally:
        conn.close()
    assert after == real_message_count, (
        f"测试污染了真实索引库：messages 由 {real_message_count} 变为 {after}"
    )


def test_new_clear_and_switch_conversation(engine, requires_index) -> None:
    """会话的新建、清空、切换与非法切换的行为契约。"""
    conversation_id = engine.new_conversation(title="会话管理测试")
    assert conversation_id, "new_conversation 未返回会话 ID"
    assert any(item.conversation_id == conversation_id for item in engine.list_conversations()), (
        "新建的会话未出现在会话列表中"
    )

    engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", conversation_id=conversation_id)
    assert len(engine.get_messages(conversation_id)) == 2, "会话内消息数不正确"

    removed = engine.clear_conversation(conversation_id)
    assert removed == 2, f"清空会话应删除 2 条消息，实际 {removed}"
    assert engine.get_messages(conversation_id) == [], "清空后仍能读到消息"

    engine.switch_conversation(conversation_id)  # 存在的会话不应抛异常
    with pytest.raises(KeyError):
        engine.switch_conversation("conv_不存在的会话")


def test_submit_feedback(engine, requires_index) -> None:
    """反馈接口：非法评分必须报错，合法反馈必须落库并可统计。"""
    conversation_id = engine.new_conversation(title="反馈测试")
    engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？", conversation_id=conversation_id)
    message_id = engine.get_messages(conversation_id)[-1].message_id

    feedback_id = engine.submit_feedback(conversation_id, message_id, "up", "回答准确", "注册资本是多少？")
    assert feedback_id > 0, "反馈未写入数据库"
    assert engine.store.feedback_stats()["up"] >= 1, "点赞数未统计到"

    with pytest.raises(ValueError, match="up 或 down"):
        engine.submit_feedback(conversation_id, message_id, "maybe")
