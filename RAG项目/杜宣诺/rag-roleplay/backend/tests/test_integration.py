"""端到端集成测试（需本机 Docker 引擎运行）。

拉起 testcontainers MySQL + Redis，跑「注册 → 建角色 → 建会话 → 对话」链路。
LLM / Embedding / Rerank 仍用 fake，不访问外部模型服务。
"""

import shutil
import subprocess

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base, get_db
from app.main import create_app


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="本机 Docker 引擎未运行，跳过集成测试"
)


@pytest.fixture(scope="module")
def mysql():
    from testcontainers.community.mysql import MySqlContainer

    with MySqlContainer("mysql:8") as c:
        # 强制使用 asyncmy 异步驱动（testcontainers 默认 URL 驱动名可能为空或 pymysql）
        url = make_url(c.get_connection_url()).set(drivername="mysql+asyncmy")
        yield url


@pytest.fixture(scope="module")
def redis_url():
    from testcontainers.community.redis import RedisContainer

    with RedisContainer("redis:7") as c:
        yield f"redis://{c.get_container_host_ip()}:{c.get_exposed_port(6379)}/0"


@pytest.fixture
async def client(mysql, redis_url, monkeypatch):
    from app.core.fakes import FakeEmbedding, FakeLLM, FakeRerank

    engine = create_async_engine(mysql, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with factory() as s:
            yield s

    import app.api.routes.chat as chat_mod
    monkeypatch.setattr(chat_mod, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(chat_mod, "get_embedding", lambda: FakeEmbedding())
    monkeypatch.setattr(chat_mod, "get_rerank", lambda: FakeRerank())
    monkeypatch.setattr(chat_mod, "get_milvus", lambda: _NoopMilvus())
    monkeypatch.setattr(chat_mod, "get_redis", lambda: _NoopRedis())

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await engine.dispose()


class _NoopRedis:
    def __init__(self):
        self.history = []

    async def get_summary(self, sid):
        return None

    async def get_recent(self, sid, rounds):
        return list(self.history[-2 * rounds:])

    async def push_message(self, sid, role, content):
        self.history.append({"role": role, "content": content})


class _NoopMilvus:
    async def hybrid_search(self, *a, **k):
        return []


async def test_full_chat_flow(client):
    await client.post("/api/v1/auth/register", json={"username": "alice", "password": "secret"})
    login = await client.post("/api/v1/auth/login", json={"username": "alice", "password": "secret"})
    token = login.json()["data"]["token"]
    h = {"Authorization": f"Bearer {token}"}

    cid = (await client.post("/api/v1/characters", json={"name": "小白", "greeting": "你好！"}, headers=h)).json()["data"]["id"]
    sid = (await client.post("/api/v1/sessions", json={"character_id": cid}, headers=h)).json()["data"]["id"]

    r = await client.post("/api/v1/chat", json={"session_id": sid, "content": "你好"}, headers=h)
    assert r.status_code == 200
    assert r.json()["data"]["reply"] == "echo: 你好"

    msgs = (await client.get(f"/api/v1/sessions/{sid}/messages", headers=h)).json()["data"]
    assert msgs[0]["content"] == "你好！"
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "echo: 你好"
