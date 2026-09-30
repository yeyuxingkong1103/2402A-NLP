# 导入测试客户端，用于模拟 HTTP 请求
import pytest
from fastapi.testclient import TestClient

# 导入 FastAPI 应用实例
from app.main import app


# 创建测试客户端 fixture
@pytest.fixture(scope="module")
def client():
    """创建测试客户端并注入内存认证存储（测试不得连真实 MySQL/Redis）。"""
    import app.auth.router as auth_router
    from app.auth.mailer import InMemoryMailer
    from app.auth.service import AuthService, InMemoryAuthStore
    from app.auth.session_store import SessionStore
    from tests.conftest import FakeRedis

    # 备份并替换服务单例：用户主体用内存存储，会话令牌用 FakeRedis
    original_service = auth_router._auth_service
    auth_router._auth_service = AuthService(
        store=InMemoryAuthStore(),
        mailer=InMemoryMailer(),
        session_store=SessionStore(FakeRedis(), 3600),
    )
    with TestClient(app) as test_client:
        yield test_client
    # 恢复原状，避免污染其他测试模块
    auth_router._auth_service = original_service


def test_send_register_code_returns_correct_format(client) -> None:
    """测试发送注册验证码接口返回正确的响应格式。"""
    # 发送请求
    response = client.post(
        "/api/v1/auth/register/code",
        json={"email": "test@qq.com", "purpose": "register"},
    )

    # 验证响应状态码
    assert response.status_code == 200

    # 验证响应体结构
    data = response.json()
    assert "code" in data
    assert "message" in data
    assert "data" in data
    assert "request_id" in data

    # 验证成功响应的字段值
    assert data["code"] == 0
    assert data["message"] == "验证码已发送"
    assert data["data"]["status"] == "sent"

    # 验证不回显验证码
    assert "code" not in data["data"]
    assert "debug_code" not in data["data"]


def test_send_reset_code_returns_correct_format(client) -> None:
    """测试发送密码重置验证码接口返回正确的响应格式。"""
    # 发送请求
    response = client.post(
        "/api/v1/auth/password-reset/code",
        json={"email": "test@qq.com", "purpose": "reset"},
    )

    # 验证响应状态码
    assert response.status_code == 200

    # 验证响应体结构
    data = response.json()
    assert "code" in data
    assert "message" in data
    assert "data" in data
    assert "request_id" in data

    # 验证成功响应的字段值
    assert data["code"] == 0
    assert data["message"] == "验证码已发送"
    assert data["data"]["status"] == "sent"

    # 验证不回显验证码
    assert "code" not in data["data"]


def test_invalid_email_domain_returns_error(client) -> None:
    """测试不支持的邮箱域名返回错误。"""
    # 发送请求（使用不支持的邮箱域名）
    response = client.post(
        "/api/v1/auth/register/code",
        json={"email": "test@gmail.com", "purpose": "register"},
    )

    # 验证返回 422 验证错误
    assert response.status_code == 422


def test_register_with_valid_code_returns_token(client) -> None:
    """测试使用有效验证码注册返回访问令牌。"""
    # 导入认证服务以获取测试环境中的验证码
    from app.auth.router import _get_auth_service

    # 先发送验证码
    client.post(
        "/api/v1/auth/register/code",
        json={"email": "newuser@qq.com", "purpose": "register"},
    )

    # 从内存存储中获取实际的验证码（仅测试环境）
    service = _get_auth_service()
    code, _ = service.store.codes[("newuser@qq.com", "register")]

    # 使用实际验证码注册
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "newuser@qq.com", "password": "password123", "code": code},
    )

    # 验证响应状态码
    assert response.status_code == 200

    # 验证响应体结构
    data = response.json()
    assert data["code"] == 0
    assert data["message"] == "注册成功"
    assert "access_token" in data["data"]
    assert data["data"]["token_type"] == "bearer"


def test_login_with_correct_password_returns_token(client) -> None:
    """测试使用正确密码登录返回访问令牌。"""
    # 导入认证服务以获取测试环境中的验证码
    from app.auth.router import _get_auth_service

    # 先注册用户
    client.post(
        "/api/v1/auth/register/code",
        json={"email": "loginuser@qq.com", "purpose": "register"},
    )

    # 获取实际的验证码
    service = _get_auth_service()
    code, _ = service.store.codes[("loginuser@qq.com", "register")]

    client.post(
        "/api/v1/auth/register",
        json={"email": "loginuser@qq.com", "password": "password123", "code": code},
    )

    # 登录
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "loginuser@qq.com", "password": "password123"},
    )

    # 验证响应状态码
    assert response.status_code == 200

    # 验证响应体结构
    data = response.json()
    assert data["code"] == 0
    assert data["message"] == "登录成功"
    assert "access_token" in data["data"]


def test_login_with_wrong_password_returns_error(client) -> None:
    """测试使用错误密码登录返回错误。"""
    # 导入认证服务以获取测试环境中的验证码
    from app.auth.router import _get_auth_service

    # 先注册用户
    client.post(
        "/api/v1/auth/register/code",
        json={"email": "wrongpwd@qq.com", "purpose": "register"},
    )

    # 获取实际的验证码
    service = _get_auth_service()
    code, _ = service.store.codes[("wrongpwd@qq.com", "register")]

    client.post(
        "/api/v1/auth/register",
        json={"email": "wrongpwd@qq.com", "password": "password123", "code": code},
    )

    # 使用错误密码登录
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "wrongpwd@qq.com", "password": "wrongpassword"},
    )

    # 验证返回 401 未认证错误
    assert response.status_code == 401

    # 验证错误响应格式
    data = response.json()
    assert data["code"] == 40100
    assert "message" in data
    assert data["data"] is None


def test_health_check_endpoints_return_correct_format(client) -> None:
    """测试健康检查端点返回正确的响应格式。"""
    # 测试存活探针
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["message"] == "success"
    assert data["data"]["status"] == "alive"
    assert "request_id" in data

    # 测试就绪探针
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["message"] == "success"
    assert data["data"]["status"] == "ready"
    assert "request_id" in data


def test_register_binds_real_user_id_not_pending(client) -> None:
    """测试注册后会话绑定真实 user_id，不得出现 'pending' 占位值。"""
    from app.auth.router import _get_auth_service

    # 发送验证码
    client.post(
        "/api/v1/auth/register/code",
        json={"email": "realid@qq.com"},
    )

    # 获取验证码
    service = _get_auth_service()
    code, _ = service.store.codes[("realid@qq.com", "register")]

    # 注册
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "realid@qq.com", "password": "password123", "code": code},
    )

    assert response.status_code == 200
    token = response.json()["data"]["access_token"]

    # 验证会话绑定真实 user_id
    session_user = service.session_store.read_session(token)
    assert session_user is not None
    assert session_user.user_id != "pending"
    assert len(session_user.user_id) > 0

    # 验证 user_sessions 键使用真实 user_id
    user_sessions_key = f"user_sessions:{session_user.user_id}"
    assert token in service.session_store.redis.smembers(user_sessions_key)

    # 验证不存在 "pending" 占位键（检查 Redis 键名）
    pending_key = "user_sessions:pending"
    assert len(service.session_store.redis.smembers(pending_key)) == 0


def test_logout_requires_valid_token(client) -> None:
    """测试 logout 必须校验会话：失效/伪造令牌返回 40100。"""
    # 伪造令牌注销
    response = client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": "Bearer fake-token-12345"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == 40100
