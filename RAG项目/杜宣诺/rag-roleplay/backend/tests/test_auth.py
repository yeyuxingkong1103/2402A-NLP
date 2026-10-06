async def test_register_and_login(client):
    r = await client.post("/api/v1/auth/register", json={"username": "alice", "password": "secret"})
    assert r.status_code == 200
    assert r.json()["code"] == 0
    r2 = await client.post("/api/v1/auth/login", json={"username": "alice", "password": "secret"})
    assert r2.status_code == 200
    assert "token" in r2.json()["data"]


async def test_me_requires_token(client):
    r = await client.get("/api/v1/users/me")
    assert r.status_code == 401


async def test_me_returns_user(client):
    await client.post("/api/v1/auth/register", json={"username": "bob", "password": "pw"})
    login = await client.post("/api/v1/auth/login", json={"username": "bob", "password": "pw"})
    token = login.json()["data"]["token"]
    r = await client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["data"]["username"] == "bob"
