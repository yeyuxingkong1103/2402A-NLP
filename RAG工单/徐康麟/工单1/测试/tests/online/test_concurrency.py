"""在线测试：并发稳定性（工单 9.2 / 验收标准 7）。

测试目标：
1. 5~10 个并发线程同时调用 ``QAEngine.ask`` 时不得抛异常；
2. 同一问题的并发答案必须一致（不存在线程间状态串扰）；
3. 并发写入 SQLite（对话/消息）时不得出现 database is locked 等锁错误；
4. 不同会话并发写入必须各自完整落库，不串号、不丢消息；
5. 并发场景下的首字延迟仍应满足 3 秒预算的量级要求。

实现方式：``concurrent.futures.ThreadPoolExecutor``（与 Streamlit 的多线程模型一致）。

索引缺失时整个模块跳过：
``索引未就绪：请先运行 python scripts/build_index.py``。
"""

from __future__ import annotations

import concurrent.futures as futures
import time

import pytest

QUESTION_CAPITAL = "武汉兴图新科电子股份有限公司注册资本是多少？"
QUESTION_LEGAL = "武汉兴图新科电子股份有限公司法定代表人是谁？"
QUESTION_REVENUE = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"

# 并发规模：工单要求模拟 5~10 并发
THREADS = 8
# 单个任务的超时（秒），超过说明链路出现阻塞
TASK_TIMEOUT_S = 120


def _run_concurrently(worker, count: int) -> tuple[list, list[str]]:
    """并发执行 ``worker(i)``，返回 ``(结果列表, 异常描述列表)``。"""
    results: list = []
    errors: list[str] = []
    with futures.ThreadPoolExecutor(max_workers=count) as pool:
        submitted = [pool.submit(worker, index) for index in range(count)]
        for task in futures.as_completed(submitted, timeout=TASK_TIMEOUT_S):
            try:
                results.append(task.result(timeout=TASK_TIMEOUT_S))
            except Exception as exc:  # noqa: BLE001 - 需要把任何异常原样汇报出来
                errors.append(f"{type(exc).__name__}: {exc}")
    return results, errors


def test_concurrent_same_question_same_conversation(engine, requires_index) -> None:
    """8 个线程在同一会话并发问同一个问题：答案必须完全一致且无异常。"""
    conversation_id = engine.new_conversation(title="并发-同会话")

    def worker(_index: int) -> str:
        return engine.ask(QUESTION_CAPITAL, conversation_id=conversation_id, save=True).answer

    started = time.perf_counter()
    answers, errors = _run_concurrently(worker, THREADS)
    elapsed = time.perf_counter() - started

    assert not errors, f"并发调用出现异常：{errors}"
    assert len(answers) == THREADS, f"应返回 {THREADS} 个答案，实际 {len(answers)}"
    assert len(set(answers)) == 1, f"同一问题在并发下答案不一致：{sorted(set(answers))}"
    assert elapsed < TASK_TIMEOUT_S, f"并发耗时 {elapsed:.1f}s，链路存在阻塞"

    messages = engine.get_messages(conversation_id)
    assert len(messages) == THREADS * 2, (
        f"并发问答应落库 {THREADS * 2} 条消息（问+答），实际 {len(messages)} 条，存在写入丢失"
    )


def test_concurrent_mixed_questions_are_consistent(engine, requires_index) -> None:
    """3 类问题各 3 个并发：同一问题的答案必须一致，不同问题的答案必须不同。"""
    questions = [QUESTION_CAPITAL, QUESTION_LEGAL, QUESTION_REVENUE]

    def worker(index: int) -> tuple[str, str]:
        question = questions[index % len(questions)]
        answer = engine.ask(question, save=False)
        return question, answer.answer

    results, errors = _run_concurrently(worker, len(questions) * 3)
    assert not errors, f"并发调用出现异常：{errors}"

    grouped: dict[str, set[str]] = {}
    for question, answer in results:
        grouped.setdefault(question, set()).add(answer)

    for question, answers in grouped.items():
        assert len(answers) == 1, f"问题『{question[:20]}』在并发下出现 {len(answers)} 种答案：{answers}"
    assert len({next(iter(values)) for values in grouped.values()}) == len(questions), (
        "不同问题的答案完全相同，检索/生成存在状态串扰"
    )


def test_concurrent_distinct_conversations_are_isolated(engine, requires_index) -> None:
    """每个线程使用独立会话：消息必须各自完整落库，不得串号。"""
    conversations = [engine.new_conversation(title=f"并发-独立-{index}") for index in range(THREADS)]

    def worker(index: int) -> int:
        engine.ask(QUESTION_LEGAL, conversation_id=conversations[index], save=True)
        return len(engine.get_messages(conversations[index]))

    counts, errors = _run_concurrently(worker, THREADS)
    assert not errors, f"并发调用出现异常：{errors}"
    assert all(count == 2 for count in counts), f"各会话消息数应为 2，实际 {counts}"


def test_concurrent_reads_and_writes_do_not_lock(engine, requires_index) -> None:
    """读写混合并发：不得出现 database is locked 等 SQLite 锁错误。"""

    def worker(index: int) -> str:
        if index % 3 == 0:
            engine.list_conversations()
            return "read-conversations"
        if index % 3 == 1:
            conversation_id = engine.new_conversation(title=f"并发-读写-{index}")
            engine.ask(QUESTION_CAPITAL, conversation_id=conversation_id, save=True)
            return "write"
        engine.stats()
        return "read-stats"

    _, errors = _run_concurrently(worker, 9)
    joined = " | ".join(errors)
    assert not errors, f"读写并发出现异常（疑似数据库锁）：{joined}"
    assert "locked" not in joined.lower(), f"出现 SQLite 锁错误：{joined}"


def test_concurrent_latency_stays_within_budget(engine, requires_index) -> None:
    """并发压力下每次调用的首字延迟仍必须满足 3 秒预算。"""

    def worker(_index: int) -> float:
        return engine.ask(QUESTION_REVENUE, save=False).first_token_ms

    timings, errors = _run_concurrently(worker, THREADS)
    assert not errors, f"并发调用出现异常：{errors}"
    assert timings, "并发调用没有返回任何延迟数据"

    worst = max(timings)
    assert worst < 3000, f"并发场景下最差首字延迟 {worst:.1f}ms 超过 3000ms 预算（全部数据：{timings}）"
