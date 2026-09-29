# -*- coding: utf-8 -*-
"""存储层测试：Redis 短期记忆 / 种子数据与模板升级 / 用户密码哈希。"""

import os
import pytest
import redis
from app.store.memory import RedisMemoryStore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.models.db import Base
from app.models.tables import Role
from app.seed import (
    LEGACY_DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    seed_roles,
    upgrade_default_prompt_templates,
)
from app.models.tables import User

# ---------- Redis 短期记忆（原 tests/test_memory.py） ----------

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
TEST_DB = 15


@pytest.fixture()
def store():
    client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=TEST_DB
    )
    client.ping()  # 连不上直接报错，而不是静默失败
    client.flushdb()
    memory = RedisMemoryStore(client, max_rounds=10, ttl_seconds=3600)
    yield memory
    client.flushdb()
    client.close()


def test_append_and_get_history_returns_turns_in_order(store):
    store.append(user_id=1, role_id=1, sender="user", content="你好")
    store.append(user_id=1, role_id=1, sender="assistant", content="你好呀")

    history = store.get_history(user_id=1, role_id=1)

    assert history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]


def test_get_history_returns_empty_list_for_new_session(store):
    assert store.get_history(user_id=1, role_id=1) == []


def test_history_is_truncated_to_max_rounds(store):
    for i in range(12):  # 12 轮 = 24 条消息，只应保留最近 10 轮
        store.append(user_id=1, role_id=1, sender="user", content=f"问{i}")
        store.append(user_id=1, role_id=1, sender="assistant", content=f"答{i}")

    history = store.get_history(user_id=1, role_id=1)

    assert len(history) == 20
    assert history[0] == {"role": "user", "content": "问2"}
    assert history[-1] == {"role": "assistant", "content": "答11"}


def test_clear_removes_session_history(store):
    store.append(user_id=1, role_id=1, sender="user", content="你好")
    store.clear(user_id=1, role_id=1)
    assert store.get_history(user_id=1, role_id=1) == []


def test_sessions_are_isolated_by_user_and_role(store):
    store.append(user_id=1, role_id=1, sender="user", content="A的会话")
    store.append(user_id=2, role_id=1, sender="user", content="B的会话")

    assert store.get_history(user_id=1, role_id=2) == []
    assert store.get_history(user_id=2, role_id=2) == []
    assert store.get_history(user_id=2, role_id=1) == [
        {"role": "user", "content": "B的会话"}
    ]

# ---------- 种子与模板升级（原 tests/test_seed.py） ----------

def make_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()


def test_seed_roles_inserts_missing_roles_only():
    session = make_session()
    added = seed_roles(session)
    assert added == 5
    assert session.query(Role).count() == 5
    # 再次执行不重复插入
    assert seed_roles(session) == 0
    assert session.query(Role).count() == 5


def test_seed_roles_use_current_template_with_knowledge():
    session = make_session()
    seed_roles(session)
    role = session.query(Role).first()
    assert "{knowledge}" in role.prompt_template


def test_upgrade_replaces_legacy_default_template():
    session = make_session()
    legacy = Role(
        name="旧角色",
        category="医生",
        persona="x",
        prompt_template=LEGACY_DEFAULT_PROMPT_TEMPLATE,
    )
    custom = Role(
        name="自定义角色",
        category="客服",
        persona="x",
        prompt_template="我自定义的模板 {user_input} 不升级",
    )
    session.add_all([legacy, custom])
    session.commit()

    upgraded = upgrade_default_prompt_templates(session)

    assert upgraded == 1
    assert "{knowledge}" in legacy.prompt_template
    assert custom.prompt_template == "我自定义的模板 {user_input} 不升级"


def test_upgrade_returns_zero_when_nothing_legacy():
    session = make_session()
    session.add(Role(name="新角色", category="x", persona="x", prompt_template=DEFAULT_PROMPT_TEMPLATE))
    session.commit()
    assert upgrade_default_prompt_templates(session) == 0

# ---------- 用户模型（原 tests/test_user_model.py） ----------

def test_set_password_stores_hash_not_plaintext():
    user = User(username="alice")
    user.set_password("secret123")

    assert user.password_hash != "secret123"
    assert user.verify_password("secret123")
    assert not user.verify_password("wrong")
