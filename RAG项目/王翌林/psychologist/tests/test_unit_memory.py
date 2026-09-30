"""单元测试：memory_service 短期记忆读写、MySQL 回填、摘要兜底（不触真实 Redis）。"""
from src.services import memory_service


class FakeRedisDB:
    """redis_db 的内存替身，接口与真实实现一致。"""

    def __init__(self):
        self.store: dict = {}
        self.appended: list = []

    def append_message(self, user_id, persona_id, conversation_id, role, content):
        key = (user_id, persona_id, conversation_id)
        self.store.setdefault(key, []).append({"role": role, "content": content})
        self.appended.append((key, role, content))

    def get_recent_messages(self, user_id, persona_id, conversation_id, limit=None):
        msgs = self.store.get((user_id, persona_id, conversation_id), [])
        return msgs[-limit:] if limit else msgs

    def clear_short_term(self, user_id, persona_id, conversation_id):
        self.store.pop((user_id, persona_id, conversation_id), None)

    def set_json(self, key, payload, ttl=None):
        self.store[key] = payload


def test_append_and_get_short_term(monkeypatch):
    fake = FakeRedisDB()
    monkeypatch.setattr(memory_service, "redis_db", fake)
    memory_service.append_short_term(1, 2, 3, "user", "最近睡不好")
    memory_service.append_short_term(1, 2, 3, "assistant", "愿意多说说吗")
    msgs = memory_service.get_short_term(None, 1, 2, 3)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert memory_service.short_term_turns(1, 2, 3) == 1


def test_clear_short_term(monkeypatch):
    fake = FakeRedisDB()
    monkeypatch.setattr(memory_service, "redis_db", fake)
    memory_service.append_short_term(1, 2, 3, "user", "hello")
    memory_service.clear_short_term(1, 2, 3)
    assert fake.store == {}


def test_short_term_falls_back_to_mysql(monkeypatch):
    """Redis 未命中时从 MySQL 回填并写回缓存。"""
    fake = FakeRedisDB()
    monkeypatch.setattr(memory_service, "redis_db", fake)
    monkeypatch.setattr(
        memory_service.conversation_service, "recent_messages_from_db",
        lambda db, conv_id, limit=None: [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        ],
    )
    msgs = memory_service.get_short_term(db=None, user_id=1, persona_id=2, conversation_id=3)
    assert len(msgs) == 2
    assert len(fake.appended) == 2  # 回填也写入缓存


def test_summarize_conversation_rule_fallback():
    """simple_complete 被测试环境屏蔽（返回空串）→ 走规则兜底摘要。"""
    messages = [
        {"role": "user", "content": "我最近失眠"},
        {"role": "assistant", "content": "嗯，谈谈具体情形"},
        {"role": "user", "content": "工作压力太大"},
    ]
    summary = memory_service.summarize_conversation(messages)
    assert "失眠" in summary and "工作压力太大" in summary


def test_summarize_empty_returns_empty():
    assert memory_service.summarize_conversation([]) == ""
