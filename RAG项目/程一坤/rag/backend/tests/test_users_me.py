"""批次 16-A：GET /api/v1/users/me（接口文档 3.8）测试。

验收点：
- 未登录 → 401（40100）
- 管理员 → 200，is_admin=true
- 普通用户 → 200，is_admin=false（不因"没有 admin 数据"报错）
- 只返回本人信息；不返回 password_hash 等敏感字段
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

ADMIN_KEY = "a" * 32
NORMAL_KEY = "b" * 32


@pytest.fixture()
def users_env():
    """SQLite 会话工厂（users 表）+ 可切换的认证身份。"""
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
                user_key=ADMIN_KEY,
                email="admin@test.local",
                password_hash="x" * 64,
                is_admin=True,
                long_term_memory_enabled=True,
            )
        )
        session.add(
            User(
                user_key=NORMAL_KEY,
                email="normal@test.local",
                password_hash="x" * 64,
                is_admin=False,
                long_term_memory_enabled=False,
            )
        )
        session.commit()
    app.state.users_me_session_factory = factory
    yield {"factory": factory}
    app.state.__dict__.pop("users_me_session_factory", None)
    app.dependency_overrides.pop(get_current_user, None)


def test_users_me_requires_login() -> None:
    """未登录 → 401（40100）。"""
    app.dependency_overrides.pop(get_current_user, None)
    with TestClient(app) as client:
        response = client.get("/api/v1/users/me")
    assert response.status_code == 401
    assert response.json()["code"] == 40100


def test_users_me_admin_returns_fields(users_env) -> None:
    """管理员 → 200，四个字段齐全，不含敏感字段。"""
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id=ADMIN_KEY, is_admin=True
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/users/me")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["user_id"] == ADMIN_KEY
    assert data["email"] == "admin@test.local"
    assert data["is_admin"] is True
    assert data["long_term_memory_enabled"] is True
    assert "password_hash" not in data
    assert "access_token" not in data


def test_users_me_normal_user_gets_false_without_error(users_env) -> None:
    """普通用户 → 200，is_admin=false，不因无 admin 数据报错。"""
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id=NORMAL_KEY, is_admin=False
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/users/me")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["is_admin"] is False
    assert data["long_term_memory_enabled"] is False
    assert data["email"] == "normal@test.local"


def test_users_me_session_user_missing_row(users_env) -> None:
    """会话有效但 users 行已不存在（边缘）：仍 200，敏感信息不外泄。"""
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id="c" * 32, is_admin=False
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/users/me")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["user_id"] == "c" * 32
    assert data["email"] == ""
    assert data["is_admin"] is False
    assert data["long_term_memory_enabled"] is False


def test_users_me_ignores_user_id_param(users_env) -> None:
    """不接受 user_id 参数：传了也只返回当前登录用户本人信息。"""
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id=NORMAL_KEY, is_admin=False
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/users/me", params={"user_id": ADMIN_KEY})
    assert response.status_code == 200
    assert response.json()["data"]["user_id"] == NORMAL_KEY
