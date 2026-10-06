"""集成测试：知识库文档全生命周期（上传 → 列表 → 删除），及权限边界。

说明：不加载真实 BGE-M3、不连 Milvus——embedder 与 milvus_db 的写入/删除
在 knowledge_service 命名空间内 monkeypatch；MySQL 必须可用（文档元数据落库）。
"""
import uuid

import pytest

from src.core.config import settings


class _FakeEmbedder:
    """确定性假向量（维度与配置一致，保证入库记录结构合法）。"""

    def encode(self, texts):
        return [[0.01] * settings.embedding_dim for _ in texts]

    def encode_query(self, query):
        return [0.02] * settings.embedding_dim


@pytest.fixture
def fake_vector_backend(monkeypatch):
    from src.services import knowledge_service

    monkeypatch.setattr(knowledge_service, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(
        knowledge_service.milvus_db, "insert_chunks",
        lambda records: list(range(1, len(records) + 1)),
    )
    monkeypatch.setattr(
        knowledge_service.milvus_db, "delete_knowledge_by_doc",
        lambda doc_id, persona_id: 0,
    )


def _persona_id(personas_by_code) -> int:
    for code in ("cbt_chen", "humanistic_lin", "mindfulness_zhou"):
        if code in personas_by_code:
            return personas_by_code[code]["id"]
    pytest.skip("三个心理医生角色均未就绪，请先初始化角色")


def _upload(client, headers, persona_id: int, content: str, filename: str):
    return client.post(
        "/api/v1/knowledge/upload",
        headers=headers,
        data={"persona_id": str(persona_id), "strategy": "paragraph"},
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
    )


def test_upload_list_delete_doc_flow(client, admin_headers, personas_by_code,
                                     fake_vector_backend):
    persona_id = _persona_id(personas_by_code)
    content = "认知行为疗法认为，情绪并非由事件直接引起，而是由我们对事件的解释所决定。\n" * 20
    filename = f"pytest_kb_{uuid.uuid4().hex[:8]}.txt"

    upload = _upload(client, admin_headers, persona_id, content, filename)
    assert upload.status_code == 200, upload.text
    data = upload.json()["data"]
    doc_id = data.get("doc_id") or data.get("id")
    assert doc_id, upload.text
    assert (data.get("chunk_count") or data.get("chunks") or 0) >= 1

    docs = client.get("/api/v1/knowledge/docs", params={"persona_id": persona_id},
                      headers=admin_headers)
    assert docs.status_code == 200, docs.text
    payload = docs.json()["data"]
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    assert any(d["id"] == doc_id for d in items)

    deleted = client.delete(f"/api/v1/knowledge/docs/{doc_id}", headers=admin_headers)
    assert deleted.status_code == 200, deleted.text
    again = client.delete(f"/api/v1/knowledge/docs/{doc_id}", headers=admin_headers)
    assert again.status_code == 404


def test_upload_requires_admin(client, test_user, personas_by_code, fake_vector_backend):
    persona_id = _persona_id(personas_by_code)
    resp = _upload(client, {"Authorization": f"Bearer {test_user['access_token']}"},
                   persona_id, "普通用户上传知识库应当被拒绝。" * 10, "no-permission.txt")
    assert resp.status_code == 403


def test_delete_unknown_doc_404(client, admin_headers):
    resp = client.delete("/api/v1/knowledge/docs/99999999", headers=admin_headers)
    assert resp.status_code == 404
