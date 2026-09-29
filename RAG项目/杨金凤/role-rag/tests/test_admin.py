"""admin.py 单元测试：TestClient + mock roles/ingest/milvus_store，不连真实服务。"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin


@pytest.fixture
def client(monkeypatch):
    # mock roles / ingest / milvus_store 依赖
    roles = MagicMock()
    roles.get_role.return_value = {"collection_name": "hypertension_guide"}
    roles.list_roles.return_value = [
        {"name": "高血压医生", "collection_name": "hypertension_guide"},
        {"name": "心理医生", "collection_name": "psychology_guide"},
    ]
    monkeypatch.setattr(admin, "roles", roles)

    ingest = MagicMock()
    ingest.ingest_pdf.return_value = 42
    monkeypatch.setattr(admin, "ingest", ingest)

    ms = MagicMock()
    ms.delete_by_source = MagicMock()
    ms.list_documents.return_value = [{"source": "a.pdf", "chunk_count": 3, "created_at": 100}]
    monkeypatch.setattr(admin, "milvus_store", ms)

    app = FastAPI()
    app.include_router(admin.router)
    return TestClient(app)


def test_upload_resolves_collection_and_ingests(client):
    resp = client.post(
        "/admin/knowledge/upload",
        params={"domain": "高血压医生"},
        files={"file": ("guide.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "guide.pdf"
    assert data["collection"] == "hypertension_guide"
    assert data["chunks"] == 42
    admin.ingest.ingest_pdf.assert_called_once()


def test_upload_unknown_domain_404(client):
    admin.roles.get_role.return_value = None
    resp = client.post(
        "/admin/knowledge/upload",
        params={"domain": "不存在"},
        files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 404


def test_delete_single_collection(client):
    resp = client.delete("/admin/knowledge/a.pdf", params={"domain": "高血压医生"})
    assert resp.status_code == 200
    admin.milvus_store.delete_by_source.assert_called_once_with("hypertension_guide", "a.pdf")


def test_delete_all_collections(client):
    resp = client.delete("/admin/knowledge/a.pdf")
    assert resp.status_code == 200
    assert admin.milvus_store.delete_by_source.call_count == 2


def test_list_aggregates_across_collections(client):
    resp = client.get("/admin/knowledge/list")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["source"] == "a.pdf"
    assert data[0]["domain"] == "高血压医生"
    assert data[0]["collection"] == "hypertension_guide"
