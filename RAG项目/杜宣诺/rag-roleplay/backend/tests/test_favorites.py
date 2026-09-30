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


async def _chat_once(client, h):
    """建角色+会话+发一问，返回 (session_id, assistant_message_id)。"""
    cid = (await client.post("/api/v1/characters", json={"name": "小白"}, headers=h)).json()["data"]["id"]
    sid = (await client.post("/api/v1/sessions", json={"character_id": cid}, headers=h)).json()["data"]["id"]
    await client.post("/api/v1/chat", json={"session_id": sid, "content": "你好"}, headers=h)
    msgs = (await client.get(f"/api/v1/sessions/{sid}/messages", headers=h)).json()["data"]
    mid = [m["id"] for m in msgs if m["role"] == "assistant"][0]
    return sid, mid


async def test_favorite_create_list_delete(client, monkeypatch):
    _patch_deps(monkeypatch)
    h = await _register_and_login(client, "favvy")
    sid, mid = await _chat_once(client, h)

    # 收藏
    r = await client.post("/api/v1/favorites", json={"message_id": mid}, headers=h)
    assert r.status_code == 200
    fav_id = r.json()["data"]["id"]

    # 幂等：重复收藏返回同一 id
    r2 = await client.post("/api/v1/favorites", json={"message_id": mid}, headers=h)
    assert r2.json()["data"]["id"] == fav_id

    # messages 里 is_favorite 标记
    msgs = (await client.get(f"/api/v1/sessions/{sid}/messages", headers=h)).json()["data"]
    assert any(m["id"] == mid and m["is_favorite"] for m in msgs)

    # 列表
    favs = (await client.get("/api/v1/favorites", headers=h)).json()["data"]
    assert len(favs) == 1
    assert favs[0]["message_id"] == mid
    assert favs[0]["session_id"] == sid
    assert favs[0]["character_name"] == "小白"
    assert favs[0]["content"] == "echo: 你好"

    # 删除
    r = await client.delete(f"/api/v1/favorites/{fav_id}", headers=h)
    assert r.status_code == 200
    assert (await client.get("/api/v1/favorites", headers=h)).json()["data"] == []


async def test_favorite_cannot_target_other_users_message(client, monkeypatch):
    _patch_deps(monkeypatch)
    h1 = await _register_and_login(client, "owner")
    h2 = await _register_and_login(client, "intruder")
    _, mid = await _chat_once(client, h1)

    r = await client.post("/api/v1/favorites", json={"message_id": mid}, headers=h2)
    assert r.status_code == 404


async def test_favorite_rejects_user_message(client, monkeypatch):
    _patch_deps(monkeypatch)
    h = await _register_and_login(client, "self")
    cid = (await client.post("/api/v1/characters", json={"name": "Z"}, headers=h)).json()["data"]["id"]
    sid = (await client.post("/api/v1/sessions", json={"character_id": cid}, headers=h)).json()["data"]["id"]
    await client.post("/api/v1/chat", json={"session_id": sid, "content": "你好"}, headers=h)
    msgs = (await client.get(f"/api/v1/sessions/{sid}/messages", headers=h)).json()["data"]
    user_mid = [m["id"] for m in msgs if m["role"] == "user"][0]

    r = await client.post("/api/v1/favorites", json={"message_id": user_mid}, headers=h)
    assert r.status_code == 404
