"""pytest 公共 fixture。

设计要点：
1. TestClient 直接基于 ``src.main:app``，但**不进入 lifespan**——lifespan 会预热
   BGE-M3 / BGE-Reranker（很慢），数据库初始化已由 ``scripts/init_db.py`` 完成。
2. 临时用户通过 ``/api/v1/auth/register`` + ``/api/v1/auth/login`` 真实注册登录拿 Token。
3. MySQL / Redis / Milvus 可用性探测 fixture：服务不可用时自动 skip 对应集成用例。
4. 全局兜底屏蔽真实大模型：``llm_service.chat / chat_stream / simple_complete`` 一律
   monkeypatch，避免测试打到 DeepSeek 在线接口。
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 本地模型必须离线加载；关闭启动预热（双保险）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WARMUP_MODELS", "0")

# 统一的假大模型输出（测试可断言该文本）
STUB_ANSWER = "我听到你最近很不容易，我们先一起慢慢梳理一下这份压力。"
USERNAME_PREFIX = "pytest_"


# ==================== 真实大模型屏蔽 ====================
def _stub_chat(messages, temperature=None, max_tokens=None, model=None, timeout=None):
    return {"content": STUB_ANSWER, "tokens": 32, "finish_reason": "stop", "elapsed_ms": 1}


def _stub_chat_stream(messages, temperature=None, max_tokens=None, model=None, timeout=None):
    yield "我听到你最近很不容易，"
    yield "我们先一起慢慢梳理一下这份压力。"


def _stub_simple_complete(prompt: str, system: str = "", temperature: float = 0.3,
                          max_tokens: int = 512, timeout: float = None) -> str:
    """辅助调用（Query 改写 / 摘要 / 起标题 / 评测打分）返回空串，走规则兜底。"""
    return ""


@pytest.fixture(scope="session", autouse=True)
def block_real_llm():
    """session 级兜底：严禁测试真实调用 DeepSeek。"""
    from src.services import llm_service

    mp = pytest.MonkeyPatch()
    mp.setattr(llm_service, "chat", _stub_chat)
    mp.setattr(llm_service, "chat_stream", _stub_chat_stream)
    mp.setattr(llm_service, "simple_complete", _stub_simple_complete)
    yield mp
    mp.undo()


# ==================== 外部服务可用性探测 ====================
def _probe(func) -> bool:
    try:
        return bool(func())
    except Exception:  # 外部服务未启动
        return False


@pytest.fixture(scope="session")
def mysql_ok() -> bool:
    from src.db.mysql import health_check
    return _probe(health_check)


@pytest.fixture(scope="session")
def redis_ok() -> bool:
    from src.db import redis as redis_db
    return _probe(redis_db.health_check)


@pytest.fixture(scope="session")
def milvus_ok() -> bool:
    from src.db import milvus as milvus_db
    return _probe(milvus_db.health_check)


@pytest.fixture(scope="session")
def require_mysql(mysql_ok):
    if not mysql_ok:
        pytest.skip("MySQL 不可用，跳过集成用例")
    return True


@pytest.fixture(scope="session")
def require_redis(redis_ok):
    if not redis_ok:
        pytest.skip("Redis 不可用，跳过集成用例")
    return True


@pytest.fixture(scope="session")
def require_milvus(milvus_ok):
    if not milvus_ok:
        pytest.skip("Milvus 不可用，跳过集成用例")
    return True


# ==================== HTTP 客户端 ====================
@pytest.fixture(scope="session")
def client(require_mysql):
    """基于 src.main:app 的 TestClient（不进入 lifespan，避免模型预热）。"""
    from fastapi.testclient import TestClient

    from src.main import app

    return TestClient(app)


# ==================== 用户 / 管理员 Token ====================
def _register_and_login(client, prefix: str, password: str = "Test123456") -> dict:
    username = f"{USERNAME_PREFIX}{prefix}_{uuid.uuid4().hex[:10]}"
    resp = client.post("/api/v1/auth/register", json={
        "username": username, "password": password, "nickname": "测试用户",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["code"] == 0, resp.text

    login = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert login.status_code == 200, login.text
    data = login.json()["data"]
    return {
        "username": username,
        "password": password,
        "user_id": data["user_id"],
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "headers": {"Authorization": f"Bearer {data['access_token']}"},
    }


@pytest.fixture(scope="session")
def test_user(client, require_mysql) -> dict:
    """会话级临时普通用户。"""
    return _register_and_login(client, "main")


@pytest.fixture(scope="session")
def other_user(client, require_mysql) -> dict:
    """第二个用户，用于越权访问用例。"""
    return _register_and_login(client, "other")


@pytest.fixture
def new_user(client) -> dict:
    """函数级独立用户，避免用例之间相互影响。"""
    return _register_and_login(client, "fn")


@pytest.fixture(scope="session")
def admin_token(client, require_mysql) -> str:
    from src.core.config import settings

    resp = client.post("/api/v1/auth/login", json={
        "username": settings.admin_username, "password": settings.admin_password,
    })
    if resp.status_code != 200 or resp.json().get("code") != 0:
        pytest.skip(f"默认管理员不可用，跳过需要管理员权限的用例：{resp.text[:120]}")
    return resp.json()["data"]["access_token"]


@pytest.fixture(scope="session")
def admin_headers(admin_token) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


# ==================== 数据库会话 / 角色 ====================
@pytest.fixture
def db_session(require_mysql):
    from src.db.mysql import get_session_factory

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def personas_by_code(client, require_mysql) -> dict:
    """persona_code -> 角色详情字典。"""
    items = client.get("/api/v1/personas").json()["data"]["items"]
    result = {}
    for item in items:
        detail = client.get(f"/api/v1/personas/{item['id']}").json()["data"]
        result[item["persona_code"]] = detail
    return result