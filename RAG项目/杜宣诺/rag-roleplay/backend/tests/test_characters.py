async def _register_and_login(client, username):
    await client.post("/api/v1/auth/register", json={"username": username, "password": "pw"})
    token = (await client.post("/api/v1/auth/login", json={"username": username, "password": "pw"})).json()["data"]["token"]
    return {"Authorization": f"Bearer {token}"}


async def test_hidden_setting_only_owner(client):
    await client.post("/api/v1/auth/register", json={"username": "bob", "password": "pw"})
    token = (await client.post("/api/v1/auth/login", json={"username": "bob", "password": "pw"})).json()["data"]["token"]
    h = {"Authorization": f"Bearer {token}"}
    r = await client.post("/api/v1/characters", json={"name": "X", "hidden_setting": "秘密"}, headers=h)
    assert r.status_code == 200
    assert r.json()["data"]["hidden_setting"] == "秘密"


async def test_hidden_setting_hidden_from_others(client):
    await client.post("/api/v1/auth/register", json={"username": "bob", "password": "pw"})
    token = (await client.post("/api/v1/auth/login", json={"username": "bob", "password": "pw"})).json()["data"]["token"]
    cid = (await client.post("/api/v1/characters", json={"name": "X", "hidden_setting": "秘密"}, headers={"Authorization": f"Bearer {token}"})).json()["data"]["id"]

    await client.post("/api/v1/auth/register", json={"username": "eve", "password": "pw"})
    token2 = (await client.post("/api/v1/auth/login", json={"username": "eve", "password": "pw"})).json()["data"]["token"]
    r = await client.get(f"/api/v1/characters/{cid}", headers={"Authorization": f"Bearer {token2}"})
    assert r.status_code == 200
    assert "hidden_setting" not in r.json()["data"]


async def test_create_character_indexes_settings(client, monkeypatch):
    class _RecMilvus:
        def __init__(self):
            self.calls = []

        async def delete_by_filter(self, collection, expr):
            self.calls.append(("delete", collection, expr))

        async def upsert_settings(self, character_id, chunks):
            self.calls.append(("upsert", character_id, len(chunks)))

    rec = _RecMilvus()
    import app.api.routes.characters as characters_mod
    monkeypatch.setattr(characters_mod, "get_milvus", lambda: rec)

    h = await _register_and_login(client, "indexer")
    r = await client.post("/api/v1/characters", json={"name": "Y", "persona": "温柔"}, headers=h)
    assert r.status_code == 200
    assert any(c[0] == "delete" for c in rec.calls)
    assert any(c[0] == "upsert" for c in rec.calls)


async def test_list_characters_only_own(client):
    h = await _register_and_login(client, "listy")
    cid = (await client.post("/api/v1/characters", json={"name": "A"}, headers=h)).json()["data"]["id"]

    h2 = await _register_and_login(client, "other")
    await client.post("/api/v1/characters", json={"name": "B"}, headers=h2)

    rows = (await client.get("/api/v1/characters", headers=h)).json()["data"]
    ids = {c["id"] for c in rows}
    assert cid in ids
    assert all(c["name"] != "B" for c in rows)


async def test_update_character_owner_and_deny(client):
    h = await _register_and_login(client, "editor")
    cid = (await client.post("/api/v1/characters", json={"name": "old"}, headers=h)).json()["data"]["id"]

    r = await client.put(f"/api/v1/characters/{cid}", json={"name": "new"}, headers=h)
    assert r.status_code == 200
    assert r.json()["data"]["name"] == "new"

    h2 = await _register_and_login(client, "intruder")
    r2 = await client.put(f"/api/v1/characters/{cid}", json={"name": "hack"}, headers=h2)
    assert r2.status_code == 403


async def test_delete_character_cascades_sessions(client):
    h = await _register_and_login(client, "deleter")
    cid = (await client.post("/api/v1/characters", json={"name": "C"}, headers=h)).json()["data"]["id"]
    sid = (await client.post("/api/v1/sessions", json={"character_id": cid}, headers=h)).json()["data"]["id"]

    r = await client.delete(f"/api/v1/characters/{cid}", headers=h)
    assert r.status_code == 200

    sessions = (await client.get("/api/v1/sessions", headers=h)).json()["data"]
    assert all(s["id"] != sid for s in sessions)
    msgs = await client.get(f"/api/v1/sessions/{sid}/messages", headers=h)
    assert msgs.status_code == 404


async def test_delete_character_deny_other(client):
    h = await _register_and_login(client, "owner")
    cid = (await client.post("/api/v1/characters", json={"name": "D"}, headers=h)).json()["data"]["id"]

    h2 = await _register_and_login(client, "intruder")
    r = await client.delete(f"/api/v1/characters/{cid}", headers=h2)
    assert r.status_code == 403
