from app.core.fakes import FakeEmbedding, FakeLLM, FakeRerank


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


def _patch_deps(monkeypatch):
    import app.api.routes.chat as chat_mod
    monkeypatch.setattr(chat_mod, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(chat_mod, "get_embedding", lambda: FakeEmbedding())
    monkeypatch.setattr(chat_mod, "get_rerank", lambda: FakeRerank())
    monkeypatch.setattr(chat_mod, "get_milvus", lambda: _NoopMilvus())
    monkeypatch.setattr(chat_mod, "get_redis", lambda: _NoopRedis())


async def _register_and_login(client, username):
    await client.post("/api/v1/auth/register", json={"username": username, "password": "pw"})
    token = (await client.post("/api/v1/auth/login", json={"username": username, "password": "pw"})).json()["data"]["token"]
    return {"Authorization": f"Bearer {token}"}


async def test_chat_returns_reply(client, monkeypatch):
    _patch_deps(monkeypatch)
    h = await _register_and_login(client, "chatty")
    cid = (await client.post("/api/v1/characters", json={"name": "Z", "persona": "温柔"}, headers=h)).json()["data"]["id"]
    sid = (await client.post("/api/v1/sessions", json={"character_id": cid}, headers=h)).json()["data"]["id"]

    r = await client.post("/api/v1/chat", json={"session_id": sid, "content": "你好"}, headers=h)
    assert r.status_code == 200
    assert r.json()["data"]["reply"] == "echo: 你好"

    msgs = (await client.get(f"/api/v1/sessions/{sid}/messages", headers=h)).json()["data"]
    assert len(msgs) == 2
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "echo: 你好"
    assert "sources" in msgs[-1]


async def test_chat_stream_emits_sse(client, monkeypatch):
    _patch_deps(monkeypatch)
    h = await _register_and_login(client, "streamer")
    cid = (await client.post("/api/v1/characters", json={"name": "Z"}, headers=h)).json()["data"]["id"]
    sid = (await client.post("/api/v1/sessions", json={"character_id": cid}, headers=h)).json()["data"]["id"]

    async with client.stream("POST", "/api/v1/chat/stream", json={"session_id": sid, "content": "hi"}, headers=h) as resp:
        assert resp.status_code == 200
        body = (await resp.aread()).decode()
    assert "event: delta" in body
    assert "event: sources" in body
    assert "event: done" in body
