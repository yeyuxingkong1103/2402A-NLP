"""集成测试：注册 → 登录 → /auth/me → 刷新 Token，以及错误分支。"""
import uuid

PASSWORD = "Test123456"


def _unique_username() -> str:
    return f"pytest_auth_{uuid.uuid4().hex[:10]}"


def test_register_login_me_refresh_flow(client, require_mysql):
    username = _unique_username()

    # 1. 注册
    reg = client.post("/api/v1/auth/register", json={
        "username": username, "password": PASSWORD, "nickname": "集成测试用户",
    })
    assert reg.status_code == 200, reg.text
    body = reg.json()
    assert body["code"] == 0 and body["message"] == "注册成功"
    assert body["data"]["username"] == username
    user_id = body["data"]["user_id"]

    # 2. 重复注册 -> 409
    dup = client.post("/api/v1/auth/register", json={"username": username, "password": PASSWORD})
    assert dup.status_code == 409, dup.text
    assert dup.json()["code"] == 409
    assert "已存在" in dup.json()["message"]

    # 3. 错误密码 -> 401
    bad = client.post("/api/v1/auth/login", json={"username": username, "password": "WrongPassword"})
    assert bad.status_code == 401, bad.text
    assert bad.json()["code"] == 401

    # 4. 登录
    login = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert login.status_code == 200, login.text
    data = login.json()["data"]
    assert data["token_type"] == "bearer"
    assert data["user_id"] == user_id
    assert data["roles"] == ["user"]
    assert data["expires_in"] > 0
    headers = {"Authorization": f"Bearer {data['access_token']}"}

    # 5. /auth/me
    me = client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200, me.text
    profile = me.json()["data"]
    assert profile["username"] == username
    assert profile["nickname"] == "集成测试用户"
    assert "default_persona_id" in profile

    # 6. 刷新 Token
    refresh = client.post("/api/v1/auth/refresh", json={"refresh_token": data["refresh_token"]})
    assert refresh.status_code == 200, refresh.text
    refreshed = refresh.json()["data"]
    assert refreshed["user_id"] == user_id
    assert refreshed["access_token"]
    # 新 access token 可正常访问
    me2 = client.get("/api/v1/auth/me",
                     headers={"Authorization": f"Bearer {refreshed['access_token']}"})
    assert me2.status_code == 200


def test_login_unknown_user_401(client, require_mysql):
    resp = client.post("/api/v1/auth/login",
                       json={"username": _unique_username(), "password": PASSWORD})
    assert resp.status_code == 401
    assert resp.json()["code"] == 401


def test_me_without_token_401(client, require_mysql):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401
    assert resp.json()["code"] == 401


def test_me_with_invalid_token_401(client, require_mysql):
    resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-token"})
    assert resp.status_code == 401
    assert resp.json()["code"] == 401


def test_refresh_with_access_token_401(client, test_user):
    """access token 不能用于刷新。"""
    resp = client.post("/api/v1/auth/refresh",
                       json={"refresh_token": test_user["access_token"]})
    assert resp.status_code == 401
    assert resp.json()["code"] == 401


def test_me_with_refresh_token_401(client, test_user):
    """refresh token 不能当 access token 使用。"""
    resp = client.get("/api/v1/auth/me",
                      headers={"Authorization": f"Bearer {test_user['refresh_token']}"})
    assert resp.status_code == 401
    assert resp.json()["code"] == 401


def test_refresh_with_garbage_token_401(client, require_mysql):
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": "garbage.token.value"})
    assert resp.status_code == 401


def test_register_invalid_payload_422(client, require_mysql):
    resp = client.post("/api/v1/auth/register", json={"username": "ab", "password": "123"})
    assert resp.status_code == 422
    assert resp.json()["code"] == 422


def test_register_duplicate_email_409(client, require_mysql):
    email = f"{uuid.uuid4().hex[:8]}@example.com"
    first = client.post("/api/v1/auth/register", json={
        "username": _unique_username(), "password": PASSWORD, "email": email,
    })
    assert first.status_code == 200, first.text
    second = client.post("/api/v1/auth/register", json={
        "username": _unique_username(), "password": PASSWORD, "email": email,
    })
    assert second.status_code == 409
    assert "邮箱" in second.json()["message"]


def test_logout_revokes_token_and_audits(client, new_user, require_redis, db_session):
    """G10：logout 后当前 access token 立即失效，并写入审计日志。"""
    logout = client.post("/api/v1/auth/logout", headers=new_user["headers"])
    assert logout.status_code == 200, logout.text
    assert logout.json()["data"]["token_revoked"] is True

    # 同一 token 再访问 → 401
    me = client.get("/api/v1/auth/me", headers=new_user["headers"])
    assert me.status_code == 401
    assert "失效" in me.json()["message"]

    # 审计日志落库
    from sqlalchemy import select
    from src.models import AuditLog
    rows = db_session.execute(
        select(AuditLog).where(AuditLog.user_id == new_user["user_id"], AuditLog.action == "logout")
    ).scalars().all()
    assert rows


def test_change_password_revokes_token(client, new_user, require_redis):
    """G10：修改密码后旧 access token 失效，新密码可重新登录。"""
    resp = client.post("/api/v1/users/me/password", headers=new_user["headers"], json={
        "old_password": new_user["password"], "new_password": "NewPass456789",
    })
    assert resp.status_code == 200, resp.text

    me = client.get("/api/v1/auth/me", headers=new_user["headers"])
    assert me.status_code == 401

    relogin = client.post("/api/v1/auth/login", json={
        "username": new_user["username"], "password": "NewPass456789",
    })
    assert relogin.status_code == 200, relogin.text