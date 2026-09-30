"""Redis 存储层测试（需要本地 Redis，使用独立前缀与 db15 隔离）。"""

from __future__ import annotations

import time

import pytest

from role_rag.api.auth import hash_password

pytestmark = pytest.mark.integration


def test_session_lifecycle(isolated_redis):
    store = isolated_redis
    session_id = store.create_session("u1", "lawyer", title="测试会话")
    meta = store.session_meta(session_id)
    assert meta["user_id"] == "u1" and meta["role_id"] == "lawyer"

    store.append_message(session_id, {"role": "user", "content": "试用期多久？"})
    store.append_message(session_id, {"role": "assistant", "content": "最长六个月。[1]"})
    store.update_session(session_id, turns=1)
    assert store.message_count(session_id) == 2
    assert store.recent_context(session_id, turns=1)[0]["content"] == "试用期多久？"

    sessions = store.list_sessions("u1")
    assert len(sessions) == 1 and sessions[0]["message_count"] == 2
    assert store.session_owner(session_id) == "u1"

    store.add_citations(session_id, ["d1#0", "d1#0", "d2#0"])
    assert store.session_citations(session_id) == ["d1#0", "d2#0"]

    assert store.delete_session(session_id) is True
    assert store.session_meta(session_id) is None
    assert store.list_sessions("u1") == []


def test_summary_and_facts(isolated_redis):
    store = isolated_redis
    session_id = store.create_session("u2", "financial_planner")
    store.set_summary(session_id, "用户偏好稳健型产品，每月可投入 3000 元")
    assert "稳健型" in store.get_summary(session_id)

    store.remember_facts("u2", "financial_planner", [{"kind": "preference", "text": "用户偏好：R2"}])
    facts = store.recent_facts("u2", "financial_planner")
    assert facts and facts[0]["text"] == "用户偏好：R2"

    store.incr_profile("u2", "questions", 3)
    assert store.profile("u2")["questions"] == "3"


def test_user_seed_and_login_record(isolated_redis):
    store = isolated_redis
    created = store.seed_users(
        [{"username": "tester", "password": "tester123", "display_name": "测试",
          "roles": ["lawyer"]}],
        password_hasher=lambda value: hash_password(value, 1000),
    )
    assert created == 1
    # 幂等：再次 seed 不重复创建
    assert store.seed_users(
        [{"username": "tester", "password": "tester123", "roles": ["lawyer"]}],
        password_hasher=lambda value: hash_password(value, 1000),
    ) == 0
    user = store.get_user("tester")
    assert user["roles"] == ["lawyer"]
    public = store.list_users()
    assert "password_hash" not in public[0]


def test_retrieval_cache(isolated_redis):
    store = isolated_redis
    key = store.retrieval_cache_key("lawyer", "hybrid", 6, "试用期多久")
    assert store.cache_get(key) is None
    store.cache_set(key, {"results": [{"chunk_id": "a#0"}], "candidates": {"dense": 1}})
    cached = store.cache_get(key)
    assert cached["results"][0]["chunk_id"] == "a#0"
    store.mark_cache(True)
    store.mark_cache(False)
    assert store.cache_stats() == {"hit": 1, "miss": 1}
    assert store.clear_cache() >= 1
    assert store.cache_get(key) is None


def test_rate_limit_and_stats(isolated_redis):
    store = isolated_redis
    for _ in range(3):
        allowed, remaining = store.check_rate_limit("u3", per_minute=3)
    assert allowed is True and remaining == 0
    allowed, remaining = store.check_rate_limit("u3", per_minute=3)
    assert allowed is False

    store.incr_stat("chat_requests", 2)
    stats = store.stats()
    assert stats["total"]["chat_requests"] == 2
    assert stats["today"]["chat_requests"] == 2


def test_online_and_kb_version(isolated_redis):
    store = isolated_redis
    store.mark_online("u4")
    assert "u4" in store.online_users()
    before = store.kb_version()
    assert store.bump_kb_version() == before + 1
    assert store.describe()["prefix"] == store.prefix
