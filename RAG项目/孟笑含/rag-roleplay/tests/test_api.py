# -*- coding: utf-8 -*-
"""API 集成测试：SQLite 内存库 + 真实 Redis(15号库) + 假 LLM，覆盖注册/登录/角色/对话。"""
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import create_app
from app.models.db import Base, get_db


def make_client(fake_llm, test_redis, test_memory):
    """每个测试独立建 app：SQLite 内存库、依赖注入假 LLM 与测试 Redis。"""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    async def override_get_llm():
        return fake_llm

    def override_get_redis():
        return test_redis

    from app.api.deps import (
        get_knowledge_service, get_llm, get_memory, get_memory_service, get_redis,
    )
    from app.seed import seed_roles

    class NullKnowledgeService:
        """默认空知识库：不检索、不报错（RAG 测试用 make_client_rag 覆盖）。"""

        async def retrieve(self, *args, **kwargs):
            return []

        def list_sources(self, *args, **kwargs):
            return []

    class NullMemoryService:
        """默认空长期记忆：不检索、不沉淀（记忆链路测试另行覆盖）。"""

        def retrieve(self, *args, **kwargs):
            return []

        def remember(self, *args, **kwargs):
            pass

    app = create_app(init_prod_db=False)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm] = override_get_llm
    app.dependency_overrides[get_redis] = override_get_redis
    app.dependency_overrides[get_memory] = lambda: test_memory
    app.dependency_overrides[get_knowledge_service] = lambda: NullKnowledgeService()
    app.dependency_overrides[get_memory_service] = lambda: NullMemoryService()

    with TestSession() as db:
        seed_roles(db)

    with TestClient(app) as client:
        return client


def register(client, username="alice", password="secret123"):
    return client.post("/api/users/register", json={"username": username, "password": password})


def test_root_serves_frontend_page(fake_llm, test_redis, test_memory):
    """GET / 返回前端页面（HTML）。"""
    client = make_client(fake_llm, test_redis, test_memory)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "角色扮演" in resp.text or "chat" in resp.text.lower()


def test_frontend_static_assets_served(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_register_returns_user_and_token(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    resp = register(client)

    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "alice"
    assert body["token"]


def test_register_duplicate_username_rejected(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    register(client)
    resp = register(client)

    assert resp.status_code == 400


def test_login_wrong_password_rejected(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    register(client)
    resp = client.post("/api/users/login", json={"username": "alice", "password": "wrong"})

    assert resp.status_code == 401


def test_login_returns_token_and_token_grants_access(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    register(client)
    login = client.post("/api/users/login", json={"username": "alice", "password": "secret123"})
    token = login.json()["token"]

    ok = client.get("/api/roles", headers={"X-Token": token})
    denied = client.get("/api/roles")

    assert ok.status_code == 200
    assert denied.status_code == 401


def test_seed_roles_exist_after_startup(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]

    roles = client.get("/api/roles", headers={"X-Token": token}).json()

    names = {r["name"] for r in roles}
    assert len(roles) >= 5
    assert {"小阳", "林医生", "王律师"} <= names
    doctor = next(r for r in roles if r["name"] == "林医生")
    assert doctor["category"] == "医生"
    assert doctor["persona"]


def test_role_crud(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    headers = {"X-Token": token}

    created = client.post(
        "/api/roles",
        headers=headers,
        json={"name": "测试角色", "category": "客服", "persona": "耐心的客服"},
    )
    assert created.status_code == 200
    role_id = created.json()["id"]

    updated = client.put(
        f"/api/roles/{role_id}", headers=headers, json={"persona": "更耐心的客服"}
    )
    assert updated.status_code == 200
    assert updated.json()["persona"] == "更耐心的客服"

    deleted = client.delete(f"/api/roles/{role_id}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/api/roles", headers=headers).json()


def test_chat_returns_reply_and_persists_history(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = next(
        r["id"] for r in client.get("/api/roles", headers=headers).json() if r["name"] == "小阳"
    )

    resp = client.post("/api/chat", headers=headers, json={"role_id": role_id, "content": "你好"})

    assert resp.status_code == 200
    assert resp.json()["reply"] == "你好呀，有什么可以帮你？"

    history = client.get(f"/api/chat/history?role_id={role_id}", headers=headers).json()
    assert [m["sender"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "你好"


def test_chat_requires_existing_role(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]

    resp = client.post(
        "/api/chat", headers={"X-Token": token}, json={"role_id": 99999, "content": "你好"}
    )

    assert resp.status_code == 404


def test_chat_requires_login(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    resp = client.post("/api/chat", json={"role_id": 1, "content": "你好"})

    assert resp.status_code == 401


def test_chat_stream_returns_sse(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = next(
        r["id"] for r in client.get("/api/roles", headers=headers).json() if r["name"] == "小阳"
    )

    with client.stream(
        "POST", "/api/chat/stream", headers=headers, json={"role_id": role_id, "content": "你好"}
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())

    # SSE 块：每个 delta 一个 JSON 事件，客户端拼接即完整回复
    assert '{"delta": "你"}' in body
    assert '{"delta": "好"}' in body
    assert '{"delta": "呀"}' in body
    assert "data: [DONE]" in body


def test_history_is_isolated_between_users(fake_llm, test_redis, test_memory):
    client = make_client(fake_llm, test_redis, test_memory)
    alice = {"X-Token": register(client).json()["token"]}
    bob = {"X-Token": register(client, "bob").json()["token"]}
    role_id = client.get("/api/roles", headers=alice).json()[0]["id"]

    client.post("/api/chat", headers=alice, json={"role_id": role_id, "content": "alice 的话"})

    bob_history = client.get(f"/api/chat/history?role_id={role_id}", headers=bob).json()

    assert bob_history == []


# ---------- RAG：知识库与对话集成 ----------

class FakeEmbedder:
    def embed_documents(self, texts):
        return [[1.0] * 1024 for _ in texts]

    def embed_query(self, text):
        return [1.0] * 1024


class FakeReranker:
    def rerank(self, query, passages):
        return passages


class FakeMilvus:
    def __init__(self):
        self.collections = {}

    def ensure_collection(self, role_id):
        self.collections.setdefault(role_id, [])

    def insert_chunks(self, role_id, chunks):
        self.ensure_collection(role_id)
        self.collections[role_id].extend(chunks)

    def hybrid_search(self, role_id, query_dense, query_sparse, limit):
        return [
            c["text"] for c in self.collections.get(role_id, []) if "blood" in c["text"]
        ][:limit]

    def all_texts(self, role_id):
        return [c["text"] for c in self.collections.get(role_id, [])]

    def list_sources(self, role_id):
        sources = {}
        for c in self.collections.get(role_id, []):
            item = sources.setdefault(c["source"], {"chunks": 0, "summary": None})
            item["chunks"] += 1
            if item["summary"] is None:
                item["summary"] = c.get("summary", "")
        return [
            {"source": s, "chunks": v["chunks"], "summary": v["summary"]}
            for s, v in sources.items()
        ]

    def delete_source(self, role_id, source):
        before = len(self.collections.get(role_id, []))
        self.collections[role_id] = [
            c for c in self.collections.get(role_id, []) if c["source"] != source
        ]
        return before - len(self.collections[role_id])


def make_pdf_bytes(text: str) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    return doc.tobytes()


def make_client_rag(fake_llm, test_redis, test_memory):
    """带假知识库服务的客户端：RAG 全链路（除真实模型/Milvus）可测。"""
    client = make_client(fake_llm, test_redis, test_memory)
    from app.api.deps import get_knowledge_service
    from app.services.knowledge_service import KnowledgeService

    milvus = FakeMilvus()
    service = KnowledgeService(embedder=FakeEmbedder(), reranker=FakeReranker(), milvus=milvus)
    client.app.dependency_overrides[get_knowledge_service] = lambda: service
    return client, milvus


def test_upload_pdf_ingests_knowledge(fake_llm, test_redis, test_memory):
    client, milvus = make_client_rag(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    role_id = client.get("/api/roles", headers={"X-Token": token}).json()[0]["id"]

    resp = client.post(
        f"/api/knowledge/upload?role_id={role_id}",
        headers={"X-Token": token},
        files={"file": ("guide.pdf", make_pdf_bytes("blood pressure control guide.\n" * 10), "application/pdf")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "guide.pdf"
    assert body["chunks"] >= 1
    assert len(milvus.collections[role_id]) == body["chunks"]


def test_upload_requires_login(fake_llm, test_redis, test_memory):
    client, _ = make_client_rag(fake_llm, test_redis, test_memory)
    resp = client.post(
        "/api/knowledge/upload?role_id=1",
        files={"file": ("g.pdf", make_pdf_bytes("content. " * 10), "application/pdf")},
    )
    assert resp.status_code == 401


def test_list_and_delete_knowledge_docs(fake_llm, test_redis, test_memory):
    client, milvus = make_client_rag(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = client.get("/api/roles", headers=headers).json()[0]["id"]

    client.post(
        f"/api/knowledge/upload?role_id={role_id}", headers=headers,
        files={"file": ("a.pdf", make_pdf_bytes("blood pressure doc.\n" * 10), "application/pdf")},
    )
    docs = client.get(f"/api/knowledge/docs?role_id={role_id}", headers=headers).json()
    assert docs[0]["source"] == "a.pdf"
    assert docs[0]["chunks"] == 1
    assert docs[0]["summary"]  # 文档列表应带摘要字段

    deleted = client.delete(
        f"/api/knowledge/doc?role_id={role_id}&source=a.pdf", headers=headers
    ).json()
    assert deleted["removed"] >= 1
    assert milvus.collections[role_id] == []


def test_chat_injects_retrieved_knowledge_and_returns_sources(fake_llm, test_redis, test_memory):
    client, _ = make_client_rag(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = client.get("/api/roles", headers=headers).json()[0]["id"]

    client.post(
        f"/api/knowledge/upload?role_id={role_id}", headers=headers,
        files={"file": ("a.pdf", make_pdf_bytes("blood pressure needs low salt diet.\n" * 10), "application/pdf")},
    )
    resp = client.post(
        "/api/chat", headers=headers, json={"role_id": role_id, "content": "血压高怎么办"}
    )

    assert resp.status_code == 200
    system = fake_llm.last_messages[0]["content"]
    assert "【知识库内容】" in system
    assert "blood pressure needs low salt diet" in system
    assert resp.json()["sources"]


class RecordingMemoryService:
    """记录对话沉淀，检索时按关键词返回相关记忆（内存实现）。"""

    def __init__(self):
        self.entries = []

    def remember(self, user_id, role_id, role_name, user_input, reply):
        self.entries.append(f"用户：{user_input}\n{role_name}：{reply}")

    def retrieve(self, user_id, role_id, query, top_k=3, recall_k=10):
        return [e for e in self.entries if any(k in e for k in ["猫", "高血压"])][:top_k]


def test_longterm_memory_injected_into_second_turn(fake_llm, test_redis, test_memory):
    """第一轮对话沉淀为记忆，第二轮提示词注入【长期记忆】段。"""
    client = make_client(fake_llm, test_redis, test_memory)
    memory_svc = RecordingMemoryService()
    from app.api.deps import get_memory_service
    client.app.dependency_overrides[get_memory_service] = lambda: memory_svc

    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = client.get("/api/roles", headers=headers).json()[0]["id"]

    client.post("/api/chat", headers=headers,
                json={"role_id": role_id, "content": "我养了一只猫", "use_rag": False})
    assert len(memory_svc.entries) == 1  # 第一轮已沉淀

    resp = client.post("/api/chat", headers=headers,
                       json={"role_id": role_id, "content": "我的猫最近怎么样", "use_rag": False})
    assert resp.status_code == 200
    system = fake_llm.last_messages[0]["content"]
    assert "【长期记忆】" in system
    assert "我养了一只猫" in system


def test_longterm_memory_can_be_disabled(monkeypatch, fake_llm, test_redis, test_memory):
    """LONGTERM_MEMORY=false 时既不沉淀也不注入。"""
    from app.config import settings
    monkeypatch.setattr(settings, "longterm_memory", False)
    client = make_client(fake_llm, test_redis, test_memory)
    memory_svc = RecordingMemoryService()
    from app.api.deps import get_memory_service
    client.app.dependency_overrides[get_memory_service] = lambda: memory_svc

    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = client.get("/api/roles", headers=headers).json()[0]["id"]

    resp = client.post("/api/chat", headers=headers,
                       json={"role_id": role_id, "content": "我养了一只猫", "use_rag": False})
    assert resp.status_code == 200
    assert memory_svc.entries == []  # 未沉淀
    assert "【长期记忆】" not in fake_llm.last_messages[0]["content"]


def test_chat_returns_warnings_for_unverified_numbers(fake_llm, test_redis, test_memory):
    """回答中的数值不在检索上下文里 → warnings 字段给出提醒。"""
    client, _ = make_client_rag(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = client.get("/api/roles", headers=headers).json()[0]["id"]

    client.post(
        f"/api/knowledge/upload?role_id={role_id}", headers=headers,
        files={"file": ("a.pdf", make_pdf_bytes("blood pressure below 140.\n" * 10), "application/pdf")},
    )
    fake_llm.reply = "收缩压降到150mmHg即可。"  # 150 不在知识库里

    resp = client.post(
        "/api/chat", headers=headers,
        json={"role_id": role_id, "content": "血压控制目标？"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["warnings"], "应给出数值未依据提醒"
    assert any("150" in w for w in body["warnings"])


def test_chat_no_warnings_when_all_covered(fake_llm, test_redis, test_memory):
    client, _ = make_client_rag(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = client.get("/api/roles", headers=headers).json()[0]["id"]

    client.post(
        f"/api/knowledge/upload?role_id={role_id}", headers=headers,
        files={"file": ("a.pdf", make_pdf_bytes("blood pressure below 140.\n" * 10), "application/pdf")},
    )
    fake_llm.reply = "血压应控制在140以下。"

    resp = client.post(
        "/api/chat", headers=headers,
        json={"role_id": role_id, "content": "血压控制目标？"},
    )

    assert resp.json()["warnings"] == []


def test_chat_use_rag_false_skips_retrieval(fake_llm, test_redis, test_memory):
    client, _ = make_client_rag(fake_llm, test_redis, test_memory)
    token = register(client).json()["token"]
    headers = {"X-Token": token}
    role_id = client.get("/api/roles", headers=headers).json()[0]["id"]

    client.post(
        f"/api/knowledge/upload?role_id={role_id}", headers=headers,
        files={"file": ("a.pdf", make_pdf_bytes("blood pressure needs low salt diet.\n" * 10), "application/pdf")},
    )
    resp = client.post(
        "/api/chat", headers=headers,
        json={"role_id": role_id, "content": "血压高怎么办", "use_rag": False},
    )

    assert resp.status_code == 200
    assert "【知识库内容】" not in fake_llm.last_messages[0]["content"]
    assert resp.json()["sources"] == []
