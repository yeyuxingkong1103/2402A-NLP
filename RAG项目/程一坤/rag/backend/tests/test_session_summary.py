"""会话摘要（批次 21）：触发 / 节流 / 失败保留旧摘要 / 提示词注入 / 开关默认关。

覆盖方案文档 reports/batch20_session_summary_plan.md 第 2.5 节的验收口径：
1. 窗口未满不触发；满窗（10 轮）首次触发
2. 节流：满窗后不足 3 轮不再触发
3. 增量：再次触发时输入里带旧摘要
4. 失败路径：LLM 报错/返回空 → 旧摘要不被污染
5. 开关关闭 → 完全不读不写（行为与引入前一致）
6. 注入：提示词多出 "# 本次会话前情" 段，且位于法源清单之前
7. 持久化层：关闭不起线程；开启走后台线程写入
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from app.api.chat_persistence import persist_turn
from app.chat.prompt_builder import build_chat_messages
from app.chat.service import ChatService
from app.memory.short_term import ShortTermMemoryStore
from app.memory.summary_policy import SUMMARY_MAX_CHARS
from app.memory.summary_service import maybe_update_session_summary

USER = "user-sum"
SESSION = "session-sum"
WINDOW = 20  # 与装配侧一致：20 条 = 10 轮


class FakeRedis:
    """支持短期记忆所需命令的最小 Redis 替身（List + String + TTL）。"""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.should_raise = False

    def rpush(self, key: str, value: str) -> int:
        self._check()
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def ltrim(self, key: str, start: int, end: int) -> bool:
        self._check()
        values = self.lists.get(key, [])
        self.lists[key] = values[start:] if end == -1 else values[start : end + 1]
        return True

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        self._check()
        values = self.lists.get(key, [])
        return values[start:] if end == -1 else values[start : end + 1]

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._check()
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    def get(self, key: str) -> str | None:
        self._check()
        return self.values.get(key)

    def expire(self, key: str, seconds: int) -> bool:
        self._check()
        self.ttls[key] = seconds
        return True

    def delete(self, *keys: str) -> int:
        self._check()
        deleted = 0
        for key in keys:
            deleted += int(key in self.lists) + int(key in self.values)
            self.lists.pop(key, None)
            self.values.pop(key, None)
            self.ttls.pop(key, None)
        return deleted

    def _check(self) -> None:
        if self.should_raise:
            raise ConnectionError("模拟 Redis 连接失败")


class FakeLLM:
    """记录每次调用的 prompt，便于断言"输入里有没有旧摘要"。"""

    def __init__(self, reply: str = "用户咨询经济补偿，尚未确认是否违法解除。") -> None:
        self.reply = reply
        self.error: Exception | None = None
        self.calls: list[tuple[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.reply


class FakeRetrievalService:
    """只保留 chat 层读取前情所需的 short_term_memory 属性。"""

    def __init__(self, short_term_memory) -> None:
        self.short_term_memory = short_term_memory


def make_store(redis: FakeRedis | None = None) -> ShortTermMemoryStore:
    return ShortTermMemoryStore(redis or FakeRedis(), ttl_seconds=7200, max_messages=WINDOW)


def append_turn(store: ShortTermMemoryStore, index: int) -> None:
    """追加一轮问答（2 条消息）。"""
    store.append_message(USER, SESSION, {"role": "user", "content": f"第{index}个问题"})
    store.append_message(USER, SESSION, {"role": "assistant", "content": f"第{index}个回答"})


def run_turn(store: ShortTermMemoryStore, llm: FakeLLM, *, enabled: bool = True):
    """模拟"一轮问答结束"：先追加消息，再调用摘要服务（生产里由 persist_turn 触发）。"""
    summary = maybe_update_session_summary(
        short_term_memory=store, user_id=USER, session_id=SESSION, llm_client=llm, enabled=enabled
    )
    return summary


def prime_to_full_window(store: ShortTermMemoryStore, llm: FakeLLM) -> None:
    """跑到第 10 轮（窗口刚满）：每次追加后都调用一次摘要服务，与生产时序一致。"""
    for index in range(1, WINDOW // 2 + 1):
        append_turn(store, index)
        run_turn(store, llm)


# ---------------------------------------------------------------- 触发与节流


def test_no_summary_before_window_is_full() -> None:
    """窗口未满（前 9 轮）：最近原文都还在，压缩没有信息增量。"""
    store, llm = make_store(), FakeLLM()
    for index in range(1, WINDOW // 2):  # 1..9 轮
        append_turn(store, index)
        assert run_turn(store, llm) is None
    assert store.read_summary(USER, SESSION) is None
    assert llm.calls == []


def test_first_summary_triggers_when_window_becomes_full() -> None:
    """第 10 轮窗口刚满 → 立即产出第一份摘要。"""
    store, llm = make_store(), FakeLLM(reply="第一份摘要")
    prime_to_full_window(store, llm)

    assert store.read_summary(USER, SESSION) == "第一份摘要"
    assert len(llm.calls) == 1
    # 第一份摘要没有旧摘要可合并
    assert "已有摘要" not in llm.calls[0][1]


def test_throttle_skips_until_three_more_turns() -> None:
    """满窗后不足 3 轮不再触发（3 轮一次，把 LLM 调用压到 1/3）。"""
    store, llm = make_store(), FakeLLM()
    prime_to_full_window(store, llm)
    assert len(llm.calls) == 1

    for index in range(11, 13):  # 第 11、12 轮：距上次摘要只差 1、2 轮
        append_turn(store, index)
        assert run_turn(store, llm) is None
    assert len(llm.calls) == 1

    append_turn(store, 13)  # 第 13 轮：满 3 轮 → 再摘要一次
    assert run_turn(store, llm) == llm.reply
    assert len(llm.calls) == 2
    # 增量输入：第二次调用必须带上旧摘要一起重写
    assert "已有摘要" in llm.calls[1][1]
    assert llm.reply in llm.calls[1][1]


def test_summary_state_is_isolated_per_session_and_has_ttl() -> None:
    """节流状态按 user/session 隔离，且与会话同 TTL（不留脏数据）。"""
    redis = FakeRedis()
    store, llm = make_store(redis), FakeLLM()
    prime_to_full_window(store, llm)

    key = f"short_memory:{USER}:{SESSION}:summary_state"
    state = json.loads(redis.values[key])
    assert state == {"turns": 10, "summarized_turns": 10}
    assert redis.ttls[key] == 7200
    # 另一个会话不受影响
    assert store.read_summary_state(USER, "另一个会话") == {}


# ---------------------------------------------------------------- 失败路径


def test_llm_failure_keeps_previous_summary() -> None:
    """LLM 失败必须保留旧摘要（写空串会把积累的前情清空）。"""
    store, llm = make_store(), FakeLLM(reply="第一份摘要")
    prime_to_full_window(store, llm)

    llm.error = RuntimeError("模型超时")
    for index in range(11, 14):
        append_turn(store, index)
        run_turn(store, llm)

    assert store.read_summary(USER, SESSION) == "第一份摘要"


def test_empty_llm_reply_keeps_previous_summary() -> None:
    """模型返回空串同样不能覆盖旧摘要。"""
    store, llm = make_store(), FakeLLM(reply="第一份摘要")
    prime_to_full_window(store, llm)

    llm.reply = "   "
    for index in range(11, 14):
        append_turn(store, index)
        run_turn(store, llm)

    assert store.read_summary(USER, SESSION) == "第一份摘要"


def test_redis_failure_never_raises() -> None:
    """Redis 抖动只返回 None，绝不向调用方抛异常（不能影响问答链路）。"""
    redis = FakeRedis()
    store, llm = make_store(redis), FakeLLM()
    prime_to_full_window(store, llm)
    redis.should_raise = True

    assert run_turn(store, llm) is None


def test_overlong_summary_is_truncated() -> None:
    """超长摘要截断到 SUMMARY_MAX_CHARS，而不是原样注入撑爆提示词预算。"""
    store, llm = make_store(), FakeLLM(reply="摘" * (SUMMARY_MAX_CHARS + 120))
    prime_to_full_window(store, llm)

    assert len(store.read_summary(USER, SESSION)) == SUMMARY_MAX_CHARS


# ---------------------------------------------------------------- 开关默认关


def test_disabled_switch_reads_and_writes_nothing() -> None:
    """关闭时"不读不写"：Redis 里连 summary_state 都不该出现。"""
    redis = FakeRedis()
    store, llm = make_store(redis), FakeLLM()
    for index in range(1, 21):
        append_turn(store, index)
        assert run_turn(store, llm, enabled=False) is None

    assert llm.calls == []
    assert "summary_state" not in "".join(redis.values)
    assert store.read_summary(USER, SESSION) is None


def test_disabled_chat_service_does_not_read_summary() -> None:
    """ChatService 关闭前情时连 Redis 都不读（store 一读就报错也不影响）。"""
    redis = FakeRedis()
    store = make_store(redis)
    store.write_summary(USER, SESSION, "已有摘要")
    redis.should_raise = True

    service = ChatService(
        retrieval_service=FakeRetrievalService(store), session_summary_enabled=False
    )
    assert service._session_summary(USER, SESSION) is None

    service.session_summary_enabled = True
    assert service._session_summary(USER, SESSION) is None  # 读取失败 → 不注入，不抛错


# ---------------------------------------------------------------- 提示词注入


def test_prompt_has_no_summary_section_by_default() -> None:
    """不传前情时提示词逐字不变（引入摘要前的行为）。"""
    content = build_chat_messages("经济补偿怎么算", "# 法源清单\n[1] 劳动合同法", "1. 用户是程序员")[1]["content"]
    assert "# 本次会话前情" not in content


def test_prompt_injects_summary_before_context_block() -> None:
    """前情段必须出现在法源清单之前（清单在前、问题在后是引用铁律）。"""
    content = build_chat_messages(
        "那这个钱谁出？",
        "# 法源清单\n[1] 劳动合同法",
        None,
        "用户为企业HR，已讨论经济补偿计算；尚未确认是否违法解除。",
    )[1]["content"]
    lines = content.splitlines()
    summary_index = lines.index("# 本次会话前情（早前轮次的压缩摘要，仅供理解上下文，不得替代法源清单）")
    context_index = lines.index("# 法源清单（本次检索结果）")
    question_index = lines.index("# 用户问题")
    assert summary_index < context_index < question_index
    assert "用户为企业HR" in content


def test_chat_service_reads_summary_from_retrieval_store() -> None:
    """开启后 ChatService 能从检索侧持有的短期记忆里取到前情，超长自动截断。"""
    store = make_store()
    store.write_summary(USER, SESSION, "摘" * (SUMMARY_MAX_CHARS + 50))
    service = ChatService(
        retrieval_service=FakeRetrievalService(store), session_summary_enabled=True
    )

    assert len(service._session_summary(USER, SESSION)) == SUMMARY_MAX_CHARS
    # 缺 user/session 或检索侧没有短期记忆时都返回 None
    assert service._session_summary(None, SESSION) is None
    assert service._session_summary(USER, None) is None
    empty = ChatService(
        retrieval_service=FakeRetrievalService(None), session_summary_enabled=True
    )
    assert empty._session_summary(USER, SESSION) is None


# ---------------------------------------------------------------- 持久化层接线


def _persist(store: ShortTermMemoryStore, llm: FakeLLM) -> None:
    persist_turn(
        chat_store=None,
        short_term_memory=store,
        user_id=USER,
        session_key=SESSION,
        question="那这个钱谁出？",
        result=SimpleNamespace(answer="按经济补偿计算[1]", sources=[]),
        request_id="test-request",
    )


def test_persist_turn_does_not_touch_summary_when_disabled(monkeypatch) -> None:
    """关闭时持久化层的摘要分支整段跳过（行为与引入前一致）。"""
    import app.memory.summary_service as summary_service

    calls: list[tuple] = []
    monkeypatch.setattr(
        summary_service,
        "maybe_update_session_summary",
        lambda **kwargs: calls.append(tuple(kwargs.items())),
    )
    monkeypatch.setattr(
        "app.core.config.settings",
        SimpleNamespace(session_summary_enabled=False),
    )

    store = make_store()
    _persist(store, FakeLLM())

    time.sleep(0.1)  # 若真起了线程，这里足够让它跑完
    assert calls == []


def test_persist_turn_writes_summary_in_background_when_enabled(monkeypatch) -> None:
    """开启时走后台线程写摘要：persist_turn 立即返回，摘要稍后落地。"""
    import app.core.config as config_module
    import app.memory.summary_service as summary_service

    monkeypatch.setattr(
        config_module, "settings", SimpleNamespace(session_summary_enabled=True)
    )
    llm = FakeLLM(reply="后台线程写入的摘要")
    # 注入假 LLM：避免在测试里真的去连模型
    monkeypatch.setattr(summary_service, "_default_llm_client", llm)

    store = make_store()
    # 预备：窗口已满，且已累计 10 轮（模拟"第 11 轮结束"的持久化时刻）
    for index in range(1, WINDOW // 2 + 1):
        append_turn(store, index)
    store.write_summary_state(USER, SESSION, {"turns": 10, "summarized_turns": 0})

    _persist(store, llm)

    deadline = time.time() + 5
    while time.time() < deadline and store.read_summary(USER, SESSION) is None:
        time.sleep(0.05)
    assert store.read_summary(USER, SESSION) == "后台线程写入的摘要"
