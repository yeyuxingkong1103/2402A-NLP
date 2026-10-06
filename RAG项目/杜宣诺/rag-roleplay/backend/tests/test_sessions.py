async def _register_and_login(client, username):
    await client.post("/api/v1/auth/register", json={"username": username, "password": "pw"})
    token = (await client.post("/api/v1/auth/login", json={"username": username, "password": "pw"})).json()["data"]["token"]
    return {"Authorization": f"Bearer {token}"}


async def test_create_session_inserts_greeting(client):
    h = await _register_and_login(client, "c")
    cid = (await client.post("/api/v1/characters", json={"name": "Y", "greeting": "你好！"}, headers=h)).json()["data"]["id"]
    sid = (await client.post("/api/v1/sessions", json={"character_id": cid}, headers=h)).json()["data"]["id"]
    msgs = (await client.get(f"/api/v1/sessions/{sid}/messages", headers=h)).json()["data"]
    assert msgs[0]["content"] == "你好！"


async def test_session_ownership_enforced(client):
    h = await _register_and_login(client, "owner")
    cid = (await client.post("/api/v1/characters", json={"name": "Y"}, headers=h)).json()["data"]["id"]
    sid = (await client.post("/api/v1/sessions", json={"character_id": cid}, headers=h)).json()["data"]["id"]

    h2 = await _register_and_login(client, "intruder")
    r = await client.get(f"/api/v1/sessions/{sid}/messages", headers=h2)
    assert r.status_code == 404
