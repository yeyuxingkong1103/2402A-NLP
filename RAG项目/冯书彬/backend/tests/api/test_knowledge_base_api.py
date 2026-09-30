import pytest

from backend.app.core.security import create_access_token


@pytest.fixture(autouse=True)
def reset_knowledge_store():
    # API 测试共用真实 app，必须在用例间重置内存仓储。
    from backend.app.repositories.knowledge_base_repository import reset_store

    reset_store()


def _token(monkeypatch, user_id: str, roles: list[str]) -> str:
    # 测试 token 使用固定 HMAC 密钥，避免依赖本机环境变量。
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key-32-bytes-minimum")
    return create_access_token(user_id, "session-1", roles=roles)


def _create_reviewed_material(client, monkeypatch) -> str:
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])
    reviewer_token = _token(monkeypatch, "reviewer-1", ["content_reviewer"])
    headers = {"Authorization": f"Bearer {admin_token}"}

    whitelist_response = client.post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers=headers,
        json={
            "url": "https://example.gov.cn/laws",
            "publisher": "最高人民法院",
            "material_type": "judicial_interpretation",
        },
    )
    assert whitelist_response.status_code == 200

    snapshot_response = client.post(
        "/api/v1/knowledge-bases/crawl-snapshots",
        headers=headers,
        json={
            "source_url": "https://example.gov.cn/laws",
            "raw_html": "<html><body>司法解释正文</body></html>",
            "attachments": [],
        },
    )
    assert snapshot_response.status_code == 200

    submit_response = client.post(
        f"/api/v1/knowledge-bases/crawl-snapshots/{snapshot_response.json()['id']}/submit-review",
        headers=headers,
    )
    assert submit_response.status_code == 200

    review_response = client.post(
        f"/api/v1/knowledge-bases/materials/{submit_response.json()['id']}/review",
        headers={"Authorization": f"Bearer {reviewer_token}"},
        json={"decision": "approved", "reason": "source verified"},
    )
    assert review_response.status_code == 200
    return review_response.json()["id"]


def _create_pending_material(client, monkeypatch) -> str:
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])
    headers = {"Authorization": f"Bearer {admin_token}"}
    whitelist_response = client.post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers=headers,
        json={"url": "https://example.gov.cn/laws", "publisher": "最高人民法院", "material_type": "judicial_interpretation"},
    )
    assert whitelist_response.status_code == 200
    snapshot_response = client.post(
        "/api/v1/knowledge-bases/crawl-snapshots",
        headers=headers,
        json={"source_url": "https://example.gov.cn/laws", "raw_html": "<html>正文</html>", "attachments": []},
    )
    assert snapshot_response.status_code == 200
    material_response = client.post(
        f"/api/v1/knowledge-bases/crawl-snapshots/{snapshot_response.json()['id']}/submit-review",
        headers=headers,
    )
    assert material_response.status_code == 200
    return material_response.json()["id"]


def test_content_reviewer_can_review_but_cannot_publish(client, monkeypatch):
    material_id = _create_reviewed_material(client, monkeypatch)
    reviewer_token = _token(monkeypatch, "reviewer-1", ["content_reviewer"])

    response = client.post(
        f"/api/v1/knowledge-bases/materials/{material_id}/publish",
        headers={"Authorization": f"Bearer {reviewer_token}"},
    )

    assert response.status_code == 403


def test_super_admin_can_publish_reviewed_material(client, monkeypatch):
    material_id = _create_reviewed_material(client, monkeypatch)
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])

    response = client.post(
        f"/api/v1/knowledge-bases/materials/{material_id}/publish",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "published"
    assert response.json()["searchable"] is True


def test_super_admin_cannot_publish_pending_material(client, monkeypatch):
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])
    material_id = _create_pending_material(client, monkeypatch)

    response = client.post(
        f"/api/v1/knowledge-bases/materials/{material_id}/publish",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 409


def test_content_reviewer_cannot_create_whitelist(client, monkeypatch):
    reviewer_token = _token(monkeypatch, "reviewer-1", ["content_reviewer"])

    response = client.post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers={"Authorization": f"Bearer {reviewer_token}"},
        json={"url": "https://example.gov.cn/laws", "publisher": "最高人民法院", "material_type": "judicial_interpretation"},
    )

    assert response.status_code == 403


def test_super_admin_can_update_and_disable_whitelist(client, monkeypatch):
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])
    headers = {"Authorization": f"Bearer {admin_token}"}
    created = client.post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers=headers,
        json={"url": "https://example.gov.cn/laws", "publisher": "最高人民法院", "material_type": "judicial_interpretation"},
    )
    assert created.status_code == 200

    updated = client.patch(
        f"/api/v1/knowledge-bases/source-whitelist/{created.json()['id']}",
        headers=headers,
        json={"publisher": "最高人民检察院", "material_type": "typical_case"},
    )
    assert updated.status_code == 200
    assert updated.json()["publisher"] == "最高人民检察院"

    disabled = client.post(
        f"/api/v1/knowledge-bases/source-whitelist/{created.json()['id']}/disable",
        headers=headers,
        json={"reason": "source retired"},
    )
    assert disabled.status_code == 200
    assert disabled.json()["active"] is False


def test_content_reviewer_cannot_update_disable_or_delete_whitelist(client, monkeypatch):
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])
    reviewer_token = _token(monkeypatch, "reviewer-1", ["content_reviewer"])
    created = client.post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"url": "https://example.gov.cn/laws", "publisher": "最高人民法院", "material_type": "judicial_interpretation"},
    )
    entry_id = created.json()["id"]
    reviewer_headers = {"Authorization": f"Bearer {reviewer_token}"}

    assert client.patch(f"/api/v1/knowledge-bases/source-whitelist/{entry_id}", headers=reviewer_headers, json={"active": False}).status_code == 403
    assert client.post(f"/api/v1/knowledge-bases/source-whitelist/{entry_id}/disable", headers=reviewer_headers, json={"reason": "no access"}).status_code == 403
    assert client.delete(f"/api/v1/knowledge-bases/source-whitelist/{entry_id}", headers=reviewer_headers).status_code == 403


def test_super_admin_can_delete_whitelist(client, monkeypatch):
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])
    headers = {"Authorization": f"Bearer {admin_token}"}
    created = client.post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers=headers,
        json={"url": "https://example.gov.cn/laws", "publisher": "最高人民法院", "material_type": "judicial_interpretation"},
    )

    deleted = client.delete(f"/api/v1/knowledge-bases/source-whitelist/{created.json()['id']}", headers=headers)

    assert deleted.status_code == 200
    assert deleted.json()["id"] == created.json()["id"]


def test_reject_api_and_forbid_rejected_publish(client, monkeypatch):
    material_id = _create_pending_material(client, monkeypatch)
    reviewer_token = _token(monkeypatch, "reviewer-1", ["content_reviewer"])
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])

    rejected = client.post(
        f"/api/v1/knowledge-bases/materials/{material_id}/review",
        headers={"Authorization": f"Bearer {reviewer_token}"},
        json={"decision": "rejected", "reason": "source mismatch"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"

    publish = client.post(
        f"/api/v1/knowledge-bases/materials/{material_id}/publish",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish.status_code == 409


def test_deprecate_api_and_forbid_published_reject(client, monkeypatch):
    material_id = _create_reviewed_material(client, monkeypatch)
    admin_token = _token(monkeypatch, "admin-1", ["super_admin"])
    reviewer_token = _token(monkeypatch, "reviewer-1", ["content_reviewer"])
    published = client.post(
        f"/api/v1/knowledge-bases/materials/{material_id}/publish",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert published.status_code == 200

    late_reject = client.post(
        f"/api/v1/knowledge-bases/materials/{material_id}/review",
        headers={"Authorization": f"Bearer {reviewer_token}"},
        json={"decision": "rejected", "reason": "late reject"},
    )
    assert late_reject.status_code == 409

    deprecated = client.post(
        f"/api/v1/knowledge-bases/materials/{material_id}/deprecate",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"reason": "new version released"},
    )
    assert deprecated.status_code == 200
    assert deprecated.json()["status"] == "deprecated"
    assert deprecated.json()["searchable"] is False
