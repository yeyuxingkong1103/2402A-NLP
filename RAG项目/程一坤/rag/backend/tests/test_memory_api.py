"""长期记忆 HTTP 接口测试（批次 14，接口 8.1 / 8.2 / 8.3）。

语义：
- 未登录 → 401（CurrentUser 依赖拦截）
- 列表只返回本人记忆（user_id 取认证上下文，跨用户隔离）
- 删除：本人成功；非本人/不存在 → 404（不暴露存在性）
- 开关：PUT 后生效，关闭后不写不读（服务层已测，此处验证落库与回读）

测试用 dependency_overrides 注入会话用户，用 app.state 注入
FakeMilvus 记忆存储与 SQLite 开关存储（绝不连真实 MySQL/Milvus）。
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.current_user import get_current_user
from app.auth.session_store import SessionUser
from app.db.base import Base
from app.db.sql_models import User
from app.main import app
from tests.test_chat_memory_injection import FakeEmbedding, FakeMilvusClient
from app.memory.long_term import LongTermMemoryStore
from app.memory.memory_settings import MemorySettingsStore

USER_A = "a" * 32
USER_B = "b" * 32


@pytest.fixture()
def memory_env():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        session.add(
            User(
                user_key=USER_A,
                email="a@test.local",
                password_hash="x",
                long_term_memory_enabled=True,
            )
        )
        session.add(
            User(
                user_key=USER_B,
                email="b@test.local",
                password_hash="x",
                long_term_memory_enabled=True,
            )
        )
        session.commit()

    store = LongTermMemoryStore(
        milvus_client=FakeMilvusClient(),
        embedding_client=FakeEmbedding(),
        collection_name="legal_long_term_memory",
        dimension=24,
    )
    store.ensure_collection()

    app.state.long_term_memory = store
    app.state.memory_settings_store = MemorySettingsStore(factory)
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id=USER_A, is_admin=False
    )
    yield {"store": store, "factory": factory}
    app.dependency_overrides.pop(get_current_user, None)
    app.state.__dict__.pop("long_term_memory", None)
    app.state.__dict__.pop("memory_settings_store", None)


def client() -> TestClient:
    return TestClient(app)


def test_list_memories_requires_auth(memory_env) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    try:
        resp = client().get("/api/v1/memories")
        assert resp.status_code == 401
    finally:
        app.dependency_overrides[get_current_user] = lambda: SessionUser(
            user_id=USER_A, is_admin=False
        )


def test_list_memories_returns_only_own(memory_env) -> None:
    store = memory_env["store"]
    store.add_memory(user_id=USER_A, content="a1", summary="用户是 A 公司的程序员")
    store.add_memory(user_id=USER_B, content="b1", summary="用户在 B 公司做财务")

    resp = client().get("/api/v1/memories")

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["total"] == 1
    assert body["data"]["items"][0]["summary"] == "用户是 A 公司的程序员"
    # 列表里只有自己的记忆
    assert all("A 公司" in item["summary"] for item in body["data"]["items"])


def test_list_memories_pagination(memory_env) -> None:
    store = memory_env["store"]
    for i in range(5):
        store.add_memory(user_id=USER_A, content=f"c{i}", summary=f"记忆{i}")

    resp = client().get("/api/v1/memories?page=2&page_size=2")

    body = resp.json()
    assert body["data"]["total"] == 5
    assert len(body["data"]["items"]) == 2
    assert body["data"]["page"] == 2


def test_delete_own_memory_succeeds_and_disappears(memory_env) -> None:
    store = memory_env["store"]
    memory_id, _ = store.add_memory(user_id=USER_A, content="a1", summary="用户是 A 公司的程序员")

    resp = client().delete(f"/api/v1/memories/{memory_id}")

    assert resp.status_code == 200
    assert resp.json()["data"]["deleted"] is True
    # 删除后立即从列表与检索中消失（验收 c）
    assert client().get("/api/v1/memories").json()["data"]["total"] == 0
    assert store.search(USER_A, "程序员") == []


def test_delete_non_own_memory_returns_404(memory_env) -> None:
    store = memory_env["store"]
    memory_id, _ = store.add_memory(user_id=USER_B, content="b1", summary="用户在 B 公司做财务")

    resp = client().delete(f"/api/v1/memories/{memory_id}")

    # 非本人 → 404，不暴露存在性；原记录仍在 B 名下
    assert resp.status_code == 404
    assert resp.json()["code"] != 0
    assert len(store.list_memories(user_id=USER_B)["items"]) == 1


def test_delete_missing_memory_returns_404(memory_env) -> None:
    resp = client().delete("/api/v1/memories/no-such-id")
    assert resp.status_code == 404


def test_memory_settings_toggle_persists(memory_env) -> None:
    resp = client().put(
        "/api/v1/users/me/memory-settings",
        json={"long_term_memory_enabled": False},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["long_term_memory_enabled"] is False
    # 落库可回读（验收 d 的开关侧）
    assert memory_env["factory"] and memory_env["store"]
    settings_store = app.state.memory_settings_store
    assert settings_store.is_enabled(USER_A) is False
    # 改回
    assert settings_store.set_enabled(USER_A, True) is True
