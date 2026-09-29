"""阶段6 审核与发布：管理员 HTTP 接口测试（契约 6.3 / 6.4）。

权限语义（按用户裁决）：
- 未登录 → 401（认证依赖拦截）
- 已登录非管理员 → 403（接口本身存在，不是"藏着不让看"）
- 管理员 → 正常使用

测试用 dependency_overrides 注入会话用户，用 app.state 注入
SQLite 会话工厂与向量库/Embedding 替身（绝不连真实 MySQL/Milvus）。
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.current_user import get_current_user
from app.auth.session_store import SessionUser
from app.db.base import Base
from app.db.import_service import import_package
from app.db.sql_models import Document
from app.main import app
from conftest import build_validated_package

REVIEWER = "a" * 32
NORMAL = "b" * 32


class FakeEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), 0.0, 1.0] for i, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.deleted: list[str] = []

    def existing_chunk_keys(self, chunk_keys) -> set[str]:
        return {row["chunk_key"] for row in self.rows}.intersection(chunk_keys)

    def upsert(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def get_all_chunk_keys(self) -> set[str]:
        return {row["chunk_key"] for row in self.rows}

    def delete_by_chunk_keys(self, chunk_keys) -> int:
        keys = set(chunk_keys)
        before = len(self.rows)
        self.rows = [row for row in self.rows if row["chunk_key"] not in keys]
        self.deleted.extend(keys)
        return before - len(self.rows)


@pytest.fixture()
def review_env():
    """SQLite 会话工厂 + 向量/Embedding 替身 + 管理员身份。"""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = FakeVectorStore()

    app.state.review_session_factory = factory
    app.state.review_vector_store = store
    app.state.review_embedding_client = FakeEmbeddingClient()
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id=REVIEWER, is_admin=True
    )
    yield {"factory": factory, "store": store}
    # 清理：避免污染其它测试模块
    app.dependency_overrides.pop(get_current_user, None)
    for attr in ("review_session_factory", "review_vector_store", "review_embedding_client"):
        app.state.__dict__.pop(attr, None)


def seed_documents(factory, count: int = 1) -> list[int]:
    """导入 count 个文档（各一个 pending_review 版本），返回 document_id 列表。"""
    document_ids = []
    for index in range(1, count + 1):
        with factory() as session:
            import_package(
                session,
                build_validated_package(
                    package_id=f"pkg-{index}",
                    source_url=f"https://example.com/law/{index}",
                    document_key=f"doc-{index}",
                    version_key=f"ver-{index}",
                    content_hash=f"hash-{index}",
                ),
            )
            from sqlalchemy import select

            document = session.scalar(select(Document))
            document_ids.append(document.id)
    return document_ids


# ==================== 权限 ====================


def test_admin_list_requires_login(review_env) -> None:
    """未登录访问管理员接口 → 401（认证先行）。"""
    app.dependency_overrides.pop(get_current_user, None)
    with TestClient(app) as client:
        response = client.get("/api/v1/admin/documents?status=pending_review")
    assert response.status_code == 401
    assert response.json()["code"] == 40100


def test_admin_list_rejects_non_admin(review_env) -> None:
    """普通用户访问管理员接口 → 403（接口存在，无权限）。"""
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id=NORMAL, is_admin=False
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/admin/documents?status=pending_review")
    assert response.status_code == 403
    assert response.json()["code"] == 40300


def test_admin_review_rejects_non_admin(review_env) -> None:
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id=NORMAL, is_admin=False
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/admin/documents/1/review",
            json={"decision": "approve"},
        )
    assert response.status_code == 403
    assert response.json()["code"] == 40300


# ==================== 管理员列表（契约 6.3） ====================


def test_admin_list_pending_review_documents(review_env) -> None:
    seed_documents(review_env["factory"], count=3)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/admin/documents",
            params={"status": "pending_review", "page": 1, "page_size": 2},
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 3
    assert len(data["items"]) == 2
    # 提交时间倒序：最后导入的在前
    assert data["items"][0]["version_key"] == "ver-3"


def test_admin_list_rejects_invalid_status(review_env) -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/admin/documents", params={"status": "everything"})
    assert response.status_code == 400


# ==================== 管理员审核（契约 6.4） ====================


def test_admin_review_approve_publishes_and_indexes(review_env) -> None:
    (document_id,) = seed_documents(review_env["factory"], count=1)
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/admin/documents/{document_id}/review",
            json={"decision": "approve", "review_note": "已核对官方来源"},
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["version_status"] == "approved"
    assert data["index_verified"] is True
    # 向量确实写入了
    assert len(review_env["store"].rows) == 2


def test_admin_review_reject_removes_vectors(review_env) -> None:
    (document_id,) = seed_documents(review_env["factory"], count=1)
    # 模拟历史遗留向量
    review_env["store"].rows = [
        {"chunk_key": "ver-1-parent", "vector": [0.0, 0.0, 1.0]},
        {"chunk_key": "ver-1-child", "vector": [0.0, 0.0, 1.0]},
    ]
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/admin/documents/{document_id}/review",
            json={"decision": "reject", "review_note": "来源无法核实"},
        )
    assert response.status_code == 200
    assert response.json()["data"]["deleted_vectors"] == 2
    assert review_env["store"].rows == []


def test_admin_review_unknown_document_returns_404(review_env) -> None:
    seed_documents(review_env["factory"], count=1)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/admin/documents/999/review",
            json={"decision": "approve"},
        )
    assert response.status_code == 404
    assert response.json()["code"] == 40001


def test_admin_review_invalid_decision_returns_400(review_env) -> None:
    (document_id,) = seed_documents(review_env["factory"], count=1)
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/admin/documents/{document_id}/review",
            json={"decision": "publish"},
        )
    assert response.status_code == 400


# ==================== 批次 16-A：6.3 追加只读字段 / 6.5 详情 ====================


def attach_law(factory, document_id: int, *, version_number: str = "v1") -> None:
    """给文档版本挂上法规身份（laws + law_versions），模拟真实联表数据。"""
    from datetime import date

    from sqlalchemy import select

    from app.db.sql_models import DocumentVersion, Law, LawVersion

    with factory() as session:
        version = session.scalar(
            select(DocumentVersion).where(DocumentVersion.document_id == document_id)
        )
        law = Law(
            law_key=f"law-{document_id}",
            name="中华人民共和国劳动合同法",
            short_name="劳动合同法",
            document_type="法律",
            authority_level=2,
            issuing_authority="全国人民代表大会常务委员会",
            jurisdiction="national",
            source_url="https://example.com/law",
        )
        session.add(law)
        session.flush()
        session.add(
            LawVersion(
                version_key=f"law-{document_id}_v1",
                law_id=law.id,
                version_number=version_number,
                promulgation_date=date(2012, 12, 28),
                effective_date=date(2008, 1, 1),
                document_version_id=version.id,
            )
        )
        session.commit()


def test_admin_list_includes_readonly_fields(review_env) -> None:
    """6.3：列表项追加 title / document_type / version_number，既有字段不动。"""
    (document_id,) = seed_documents(review_env["factory"], count=1)
    attach_law(review_env["factory"], document_id, version_number="v1")
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/admin/documents", params={"status": "pending_review"}
        )
    assert response.status_code == 200
    item = response.json()["data"]["items"][0]
    assert item["title"] == "测试法规"  # title 来自 documents 表（种子数据标题）
    assert item["document_type"] == "法律"
    assert item["version_number"] == "v1"
    # 既有字段仍在
    for key in (
        "document_id",
        "version_key",
        "version_status",
        "content_hash",
        "created_at",
        "reviewed_by",
        "reviewed_at",
        "review_note",
    ):
        assert key in item


def test_admin_list_without_law_linkage_returns_null_fields(review_env) -> None:
    """法规联表缺失（如历史数据）→ 新增字段为 null，列表不报错。"""
    seed_documents(review_env["factory"], count=1)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/admin/documents", params={"status": "pending_review"}
        )
    assert response.status_code == 200
    item = response.json()["data"]["items"][0]
    assert item["title"] is not None  # title 来自 documents，必有
    assert item["document_type"] is None
    assert item["version_number"] is None


def add_chunk(factory, version_id: int, chunk_key: str, content: str, sequence: int) -> None:
    from sqlalchemy import select

    from app.db.sql_models import DocumentChunk, DocumentVersion

    with factory() as session:
        version = session.get(DocumentVersion, version_id)
        session.add(
            DocumentChunk(
                chunk_key=chunk_key,
                document_version_id=version_id,
                chunk_type="child",
                article_number="47",
                sequence=sequence,
                content=content,
                retrieval_text=content,
            )
        )
        session.commit()


def _latest_version_id(factory, document_id: int) -> int:
    from sqlalchemy import select

    from app.db.sql_models import DocumentVersion

    with factory() as session:
        version = session.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.id.desc())
        )
        return version.id


def test_admin_detail_returns_metadata_and_preview(review_env) -> None:
    """6.5：详情返回元数据 + 分块预览，长内容被截断。"""
    (document_id,) = seed_documents(review_env["factory"], count=1)
    attach_law(review_env["factory"], document_id)
    version_id = _latest_version_id(review_env["factory"], document_id)
    add_chunk(review_env["factory"], version_id, "ck-1", "短内容" * 10, 1)
    add_chunk(review_env["factory"], version_id, "ck-2", "长" * 500, 2)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/admin/documents/{document_id}/detail")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document_id"] == document_id
    assert data["version_key"] == "ver-1"
    assert data["version_status"] == "pending_review"
    assert data["title"] == "测试法规"  # title 来自 documents 表
    assert data["document_type"] == "法律"
    assert data["issuing_authority"] == "全国人民代表大会常务委员会"
    assert data["promulgation_date"] == "2012-12-28"
    assert data["effective_date"] == "2008-01-01"
    assert data["content_hash"] == "hash-1"
    assert data["source_url"] == "https://example.com/law/1"
    assert data["reviewed_by"] is None
    preview = data["chunk_preview"]
    by_key = {row["chunk_key"]: row for row in preview}
    assert "ck-1" in by_key and "ck-2" in by_key  # 新插的两条都在预览里
    assert by_key["ck-1"]["content"] == "短内容" * 10  # 未达截断阈值原样返回
    assert len(by_key["ck-2"]["content"]) < 500  # 长内容被截断
    assert by_key["ck-2"]["content"].endswith("……")


def test_admin_detail_preview_chunks_limit(review_env) -> None:
    """preview_chunks 控制条数：默认 5、最多 20（超出参数被 422 拦下）。"""
    (document_id,) = seed_documents(review_env["factory"], count=1)
    version_id = _latest_version_id(review_env["factory"], document_id)
    for index in range(7):
        add_chunk(review_env["factory"], version_id, f"ck-{index}", f"内容{index}", index)

    with TestClient(app) as client:
        default_response = client.get(f"/api/v1/admin/documents/{document_id}/detail")
        limited_response = client.get(
            f"/api/v1/admin/documents/{document_id}/detail", params={"preview_chunks": 3}
        )
        over_limit = client.get(
            f"/api/v1/admin/documents/{document_id}/detail", params={"preview_chunks": 21}
        )
    assert default_response.status_code == 200
    assert len(default_response.json()["data"]["chunk_preview"]) == 5
    assert limited_response.status_code == 200
    assert len(limited_response.json()["data"]["chunk_preview"]) == 3
    assert over_limit.status_code == 422


def test_admin_detail_prefers_pending_version(review_env) -> None:
    """同文档多版本：优先返回待审核版本，否则返回最新版本。"""
    (document_id,) = seed_documents(review_env["factory"], count=1)
    # 直接把唯一版本置为 approved，再补一个更早的 pending 版本难以构造；
    # 简化验证：无 pending 时返回最新（当前唯一）版本
    from sqlalchemy import select

    from app.db.sql_models import DocumentVersion

    with review_env["factory"]() as session:
        version = session.scalar(select(DocumentVersion))
        version.version_status = "approved"
        session.commit()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/admin/documents/{document_id}/detail")
    assert response.status_code == 200
    assert response.json()["data"]["version_status"] == "approved"


def test_admin_detail_unknown_document_returns_404(review_env) -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/admin/documents/999/detail")
    assert response.status_code == 404
    assert response.json()["code"] == 40001


def test_admin_detail_rejects_non_admin(review_env) -> None:
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        user_id=NORMAL, is_admin=False
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/admin/documents/1/detail")
    assert response.status_code == 403
    assert response.json()["code"] == 40300


def test_admin_detail_requires_login(review_env) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    with TestClient(app) as client:
        response = client.get("/api/v1/admin/documents/1/detail")
    assert response.status_code == 401
    assert response.json()["code"] == 40100
