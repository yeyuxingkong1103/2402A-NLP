import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend.app.api.deps import require_role
from backend.app.core.security import create_access_token
from backend.app.main import app


def _role_client() -> TestClient:
    # 独立应用只覆盖 require_role 依赖本身，真实路由另有测试验证。
    test_app = FastAPI()

    @test_app.post("/api/v1/knowledge-bases/source-whitelist")
    def manage_source_whitelist(_: dict = Depends(require_role("super_admin"))) -> dict[str, bool]:
        # 当前任务只验证权限依赖，不实现白名单业务。
        return {"ok": True}

    return TestClient(test_app)


def test_content_reviewer_cannot_manage_source_whitelist(monkeypatch):
    # 测试 token 使用固定 HMAC 密钥，避免依赖本机环境变量。
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    token = create_access_token("reviewer-1", "session-1", roles=["content_reviewer"])

    response = _role_client().post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers={"Authorization": f"Bearer {token}"},
        json={"url": "https://example.gov.cn/law", "publisher": "test"},
    )

    assert response.status_code == 403


def test_super_admin_can_manage_source_whitelist(monkeypatch):
    # 测试 token 使用固定 HMAC 密钥，避免依赖本机环境变量。
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    token = create_access_token("admin-1", "session-1", roles=["super_admin"])

    response = _role_client().post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers={"Authorization": f"Bearer {token}"},
        json={"url": "https://example.gov.cn/law", "publisher": "test"},
    )

    assert response.status_code == 200


def test_real_admin_route_rejects_content_reviewer(monkeypatch):
    # 使用项目真实 FastAPI app，证明生产路由已接入角色鉴权。
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    token = create_access_token("reviewer-1", "session-1", roles=["content_reviewer"])

    response = TestClient(app).get(
        "/api/v1/admin/role-check",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403


def test_real_admin_route_allows_super_admin(monkeypatch):
    # 超级管理员访问真实受保护路由应通过。
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    token = create_access_token("admin-1", "session-1", roles=["super_admin"])

    response = TestClient(app).get(
        "/api/v1/admin/role-check",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200


def test_create_access_token_rejects_unknown_role(monkeypatch):
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")

    with pytest.raises(ValueError):
        create_access_token("admin-1", "session-1", roles=["owner"])


def test_create_access_token_normalizes_role(monkeypatch):
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    token = create_access_token("admin-1", "session-1", roles=[" Super_Admin "])

    response = TestClient(app).get(
        "/api/v1/admin/role-check",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
