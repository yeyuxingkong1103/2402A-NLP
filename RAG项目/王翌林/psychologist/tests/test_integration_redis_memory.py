"""集成测试（integration）：真实 Redis 短期记忆读写、TTL、MySQL 回填。"""
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def real_db(require_mysql):
    from src.db.mysql import get_session_factory
    session = get_session_factory()()
    yield session
    session.close()


def test_append_get_clear_real_redis(require_redis, real_db):
    from src.services import memory_service

    uid, pid, cid = 999901, 1, 999901
    memory_service.clear_short_term(uid, pid, cid)  # 清理历史残留
    try:
        memory_service.append_short_term(uid, pid, cid, "user", "我最近总是失眠")
        memory_service.append_short_term(uid, pid, cid, "assistant", "这种情况持续多久了？")
        msgs = memory_service.get_short_term(real_db, uid, pid, cid)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert "失眠" in msgs[0]["content"]
        assert memory_service.short_term_turns(uid, pid, cid) == 1
    finally:
        memory_service.clear_short_term(uid, pid, cid)
    # Redis 已清空；该 cid 在 MySQL 无历史 → 回填为空
    assert memory_service.get_short_term(real_db, uid, pid, cid) == []


def test_redis_key_ttl_exists(require_redis):
    from src.core.config import settings
    from src.db import redis as redis_db

    uid, pid, cid = 999902, 1, 999902
    redis_db.append_message(uid, pid, cid, "user", "ttl 检查")
    try:
        ttl = redis_db.get_redis().ttl(redis_db.session_messages_key(uid, pid, cid))
        assert 0 < ttl <= settings.short_term_ttl
    finally:
        redis_db.clear_short_term(uid, pid, cid)


def test_limit_recent_messages(require_redis):
    from src.db import redis as redis_db

    uid, pid, cid = 999903, 1, 999903
    redis_db.clear_short_term(uid, pid, cid)
    try:
        for i in range(6):
            redis_db.append_message(uid, pid, cid, "user", f"消息{i}")
        msgs = redis_db.get_recent_messages(uid, pid, cid, limit=4)
        assert len(msgs) == 4
        assert msgs[-1]["content"] == "消息5"          # 保留最近 N 条
    finally:
        redis_db.clear_short_term(uid, pid, cid)


def test_short_term_backfill_from_mysql(require_redis, require_mysql, client, test_user, real_db):
    """清空 Redis 后 get_short_term 应从 MySQL 历史消息回填。"""
    from src.services import memory_service

    persona_id = 1
    created = client.post("/api/v1/chat", headers=test_user["headers"], json={
        "persona_id": persona_id, "message": "回填测试：我喜欢在雨夜散步。",
    })
    assert created.status_code == 200, created.text
    cid = created.json()["data"]["conversation_id"]

    memory_service.clear_short_term(test_user["user_id"], persona_id, cid)
    msgs = memory_service.get_short_term(real_db, test_user["user_id"], persona_id, cid)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "回填测试：我喜欢在雨夜散步。"
    assert msgs[1]["role"] == "assistant"
