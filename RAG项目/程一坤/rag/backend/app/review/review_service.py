"""审核与发布核心服务（阶段6）。

契约来源：docs/接口文档.md 第 6.3 / 6.4 节（6.5 详情查询在 review_detail.py）。
状态机：pending_review --approve--> approved；pending_review --reject--> rejected。
可追溯：每次审核动作记录 reviewed_by（审核人 user_key）/ reviewed_at / review_note。

与向量索引的关系（6.3"索引侧保证 + 检索侧兜底"）：
- 未审核（pending_review / rejected）版本不会被 index_awaiting_embeddings 处理，
  内容根本不进 Milvus
- approve 后本模块触发该版本的向量索引，并用 existing_chunk_keys 校验全部
  chunk 都已写入（"只有审核通过且索引校验成功的版本才会进入生效知识库"）；
  校验失败时 processing_status 停在 awaiting_embedding 供重试，不谎报 indexed
- reject 时删除该版本已存在的向量（历史遗留场景），MySQL 正文保留不删
"""

# 导入 UTC 时间（审核时间戳）
from datetime import UTC, datetime

# 导入 SQLAlchemy 查询构造器与类型
from sqlalchemy import func, select
from sqlalchemy.orm import Session

# 导入向量索引服务（approve 触发索引复用同一实现，避免两套索引逻辑）
from app.db.sql_models import Document, DocumentChunk, DocumentVersion, Law, LawVersion
from app.db.vector_index_service import index_awaiting_embeddings

# 审核状态词表来自数据层唯一定义处（本文件不再自定义字面量）
from app.db.version_status import (
    APPROVED,
    PENDING_REVIEW,
    REJECTED,
    REVIEW_STATUSES,
)

# 审核决定值域（契约 6.4）
DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"
VALID_DECISIONS = (DECISION_APPROVE, DECISION_REJECT)


class InvalidReviewDecision(ValueError):
    """decision 不是 approve/reject（API 层转 400）。"""


class NoPendingReviewVersion(ValueError):
    """该文档没有待审核版本（API 层转 404）。"""

def list_review_documents(
    session: Session, *, status: str, page: int, page_size: int
) -> dict:
    """按审核状态列出版本（提交时间倒序 + 分页），供管理员审核列表使用。

    返回：{"items": [...], "total": int, "page": int, "page_size": int}
    每条 item 含 document_id / version_key / version_status / content_hash /
    created_at / reviewed_by / reviewed_at / review_note（既有字段，语义不变），
    以及批次 16-A 追加的只读展示字段 title / document_type / version_number
    （documents + laws + law_versions 联表取得；无法规联表时为 null）。
    """
    if status not in REVIEW_STATUSES:
        raise ValueError(f"status 必须是 {'/'.join(REVIEW_STATUSES)} 之一")  # noqa: TRY003

    base = (
        select(
            DocumentVersion,
            Document.title,
            Law.document_type,
            LawVersion.version_number,
        )
        .join(Document, Document.id == DocumentVersion.document_id)
        .outerjoin(LawVersion, LawVersion.document_version_id == DocumentVersion.id)
        .outerjoin(Law, Law.id == LawVersion.law_id)
        .where(DocumentVersion.version_status == status)
    )
    total = session.scalar(
        select(func.count()).select_from(base.subquery())
    )
    rows = session.execute(
        base.order_by(DocumentVersion.created_at.desc(), DocumentVersion.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()

    items = [
        {
            "document_id": version.document_id,
            "version_key": version.version_key,
            "version_status": version.version_status,
            "content_hash": version.content_hash,
            "created_at": version.created_at.isoformat(),
            "reviewed_by": version.reviewed_by,
            "reviewed_at": version.reviewed_at.isoformat() if version.reviewed_at else None,
            "review_note": version.review_note,
            "title": title,
            "document_type": document_type,
            "version_number": law_version_number,
        }
        for version, title, document_type, law_version_number in rows
    ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}

def review_document(
    session: Session,
    *,
    document_id: int,
    reviewer_user_key: str,
    decision: str,
    review_note: str | None,
    vector_store,
    embedding_client,
    embedding_batch_size: int = 32,
) -> dict:
    """审核文档的最新待审核版本（契约 6.4：按 document_id 提交决定）。

    版本定位规则：同一文档可能积累多个 pending_review 版本（内容多次更新），
    审核对象取其中最新创建的一个（id 最大）；其余待审核版本保持不动，
    等待后续逐版审核。无待审核版本时抛 ValueError。

    参数：
    - decision: approve / reject
    - reviewer_user_key: 审核人（认证上下文的 user_key，写入 reviewed_by）

    返回：
    - dict：decision / version_key / version_status / reviewed_at +
      approve 场景附加 indexed_chunks 与 index_verified；
      reject 场景附加 deleted_vectors
    """
    if decision not in VALID_DECISIONS:
        raise InvalidReviewDecision(f"decision 必须是 {'/'.join(VALID_DECISIONS)} 之一")

    # 取该文档最新的待审核版本
    version = session.scalar(
        select(DocumentVersion)
        .where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_status == PENDING_REVIEW,
        )
        .order_by(DocumentVersion.id.desc())
    )
    if version is None:
        raise NoPendingReviewVersion("该文档没有待审核版本")

    # 审核留痕：谁、何时、什么说明（无论批准还是驳回都记录）
    version.reviewed_by = reviewer_user_key
    version.reviewed_at = datetime.now(UTC)
    version.review_note = review_note

    if decision == DECISION_APPROVE:
        return _approve(session, version, vector_store, embedding_client, embedding_batch_size)
    return _reject(session, version, vector_store)


def _approve(session: Session, version: DocumentVersion, vector_store, embedding_client, embedding_batch_size: int) -> dict:
    """批准：置状态 → 触发向量索引 → 校验向量完整性 → 切生效指针。"""
    version.version_status = APPROVED
    # 索引流程只处理 approved 且 awaiting_embedding 的版本（6.3①）；
    # 导入默认即 awaiting_embedding，这里显式置位以覆盖历史遗留状态
    version.processing_status = "awaiting_embedding"
    session.flush()

    # 触发该版本（以及其它已批准待索引版本）的向量索引
    index_result = index_awaiting_embeddings(
        session,
        embedding_client=embedding_client,
        vector_store=vector_store,
        batch_size=1000,
        embedding_batch_size=embedding_batch_size,
    )

    # 索引校验：该版本全部 chunk 必须都已写入向量库，
    # "只有审核通过且索引校验成功的版本才会进入生效知识库"（契约 6.4）
    chunk_keys = list(
        session.scalars(
            select(DocumentChunk.chunk_key).where(
                DocumentChunk.document_version_id == version.id
            )
        ).all()
    )
    existing = vector_store.existing_chunk_keys(chunk_keys)
    index_verified = set(existing) == set(chunk_keys)
    if not index_verified:
        # 校验失败：处理状态退回待索引，供重试；不谎报 indexed
        version.processing_status = "awaiting_embedding"

    # 切换生效版本指针：不删除、不改写任何历史版本内容
    document = session.get(Document, version.document_id)
    document.current_version_id = version.id
    session.commit()

    return {
        "decision": DECISION_APPROVE,
        "version_key": version.version_key,
        "version_status": version.version_status,
        "reviewed_at": version.reviewed_at.isoformat(),
        "indexed_chunks": index_result.indexed_chunks,
        "index_verified": index_verified,
    }


def _reject(session: Session, version: DocumentVersion, vector_store) -> dict:
    """驳回：置状态 + 删除该版本已存在的向量（MySQL 正文保留）。"""
    version.version_status = REJECTED
    # 该版本当前在向量库中的 chunk（历史遗留向量或此前批准后又被驳回）
    chunk_keys = list(
        session.scalars(
            select(DocumentChunk.chunk_key).where(
                DocumentChunk.document_version_id == version.id
            )
        ).all()
    )
    in_store = vector_store.existing_chunk_keys(chunk_keys)
    deleted_vectors = vector_store.delete_by_chunk_keys(list(in_store)) if in_store else 0
    session.commit()

    return {
        "decision": DECISION_REJECT,
        "version_key": version.version_key,
        "version_status": version.version_status,
        "reviewed_at": version.reviewed_at.isoformat(),
        "deleted_vectors": deleted_vectors,
    }
