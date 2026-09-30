import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db.base import Base, get_db
from app.db.models import Document, User
from app.main import create_app


class _NoopMilvus:
    def __init__(self):
        self.deleted = []

    async def delete_by_filter(self, collection, filter_expr):
        self.deleted.append((collection, filter_expr))


@pytest.fixture
async def app_client(tmp_path, monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with factory() as s:
            yield s

    import app.api.routes.documents as docs_mod
    monkeypatch.setattr(docs_mod, "get_milvus", lambda: _NoopMilvus())

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, factory, tmp_path
    await engine.dispose()


async def _login(client, username):
    await client.post("/api/v1/auth/register", json={"username": username, "password": "pw"})
    token = (await client.post("/api/v1/auth/login", json={"username": username, "password": "pw"})).json()["data"]["token"]
    me = (await client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})).json()["data"]
    return token, me["id"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


async def _make_doc(factory, user_id, filename, path, status="done"):
    async with factory() as s:
        d = Document(user_id=user_id, filename=filename, ext=filename.rsplit(".")[-1], path=str(path), status=status)
        s.add(d)
        await s.commit()
        await s.refresh(d)
        return d.id


async def test_list_is_global(app_client):
    client, factory, tmp_path = app_client
    _, uid_a = await _login(client, "doc_a")
    _, uid_b = await _login(client, "doc_b")

    p1 = tmp_path / "a.txt"
    p1.write_text("甲")
    p2 = tmp_path / "b.txt"
    p2.write_text("乙")
    await _make_doc(factory, uid_a, "a.txt", p1)
    await _make_doc(factory, uid_b, "b.txt", p2)

    # A 也能看到 B 上传的文档（全局可见）
    token_a, _ = await _login(client, "doc_a")
    rows = (await client.get("/api/v1/documents", headers=_auth(token_a))).json()["data"]
    filenames = {r["filename"] for r in rows}
    assert filenames == {"a.txt", "b.txt"}


async def test_delete_only_by_uploader(app_client):
    client, factory, tmp_path = app_client
    token_a, uid_a = await _login(client, "del_a")
    token_b, uid_b = await _login(client, "del_b")

    p = tmp_path / "secret.txt"
    p.write_text("内容")
    doc_id = await _make_doc(factory, uid_a, "secret.txt", p)

    # 非上传者删除 → 404
    r = await client.delete(f"/api/v1/documents/{doc_id}", headers=_auth(token_b))
    assert r.status_code == 404

    # 上传者删除 → 成功
    r = await client.delete(f"/api/v1/documents/{doc_id}", headers=_auth(token_a))
    assert r.status_code == 200
    assert r.json()["code"] == 0

    rows = (await client.get("/api/v1/documents", headers=_auth(token_a))).json()["data"]
    assert all(d["id"] != doc_id for d in rows)


async def test_download_returns_file(app_client):
    client, factory, tmp_path = app_client
    token_a, uid_a = await _login(client, "dl_a")
    token_b, _ = await _login(client, "dl_b")

    p = tmp_path / "note.txt"
    p.write_text("下载内容", encoding="utf-8")
    doc_id = await _make_doc(factory, uid_a, "note.txt", p)

    # 全局可见：任意登录用户均可下载
    r = await client.get(f"/api/v1/documents/{doc_id}/download", headers=_auth(token_b))
    assert r.status_code == 200
    assert r.content.decode("utf-8") == "下载内容"

    # 不存在 → 404
    r = await client.get("/api/v1/documents/999999/download", headers=_auth(token_a))
    assert r.status_code == 404


async def test_upload_rejects_oversized(app_client, monkeypatch):
    client, factory, tmp_path = app_client
    token_a, _ = await _login(client, "big_a")

    import app.api.routes.documents as docs_mod
    monkeypatch.setattr(docs_mod, "get_settings", lambda: Settings(max_upload_bytes=4, doc_upload_dir=str(tmp_path)))

    r = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("big.txt", b"12345", "text/plain")},
        headers=_auth(token_a),
    )
    assert r.status_code == 413
