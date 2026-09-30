"""审核文档详情查询（契约 6.5：版本定位 + 元数据 + 分块预览截断）。

为什么独立：这是只读查询逻辑，与 6.4 的"审核动作"（approve/reject、
向量索引触发、生效指针切换）没有任何共享状态；独立后详情接口的调整
（预览条数、截断策略）不再触碰审核写路径。

输出契约（6.5）：元数据平铺返回 + chunk_preview 列表；
正文一律截断回传（详情页只需判断"内容对不对劲"，不需要全文）。
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.sql_models import Document, DocumentChunk, DocumentVersion, Law, LawVersion
# 状态词表统一从数据层唯一定义处取（本模块是只读查询模块，不反向依赖 review_service，
# 否则两模块会循环导入；常量模块无任何 import，因此引用安全）
from app.db.version_status import PENDING_REVIEW


class ReviewDetailNotFound(ValueError):
    """审核详情目标文档不存在或没有任何版本（API 层转 404）。"""


def get_review_document_detail(
    session: Session,
    *,
    document_id: int,
    preview_chunks: int = 5,
    content_limit: int = 200,
) -> dict:
    """查询审核文档详情（契约 6.5：元数据 + 分块正文预览）。

    版本定位规则（契约 6.5）：同文档多版本时优先返回当前待审核版本，
    无待审核版本则返回最新创建的版本——version_key 字段明确是哪一版。
    分块预览按 sequence 正序取前 preview_chunks 条，内容超长做截断，
    避免整篇回传（性能与暴露面考虑）。

    preview_chunks / content_limit 由调用方（API 层）传入并做上下限校验，
    本函数只按参数执行，不做二次裁剪。
    """
    # 第一优先：该文档最新的待审核版本（管理员正在看的那一版）
    version = session.scalar(
        select(DocumentVersion)
        .where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_status == PENDING_REVIEW,
        )
        .order_by(DocumentVersion.id.desc())
    )
    # 兜底：没有待审核版本时给出最新版本（历史详情回看场景）
    if version is None:
        version = session.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.id.desc())
        )
    # 一个版本都没有 → 文档不存在（含"文档行在但版本被清理"的脏数据）
    # 注意：契约要求 404 = 40001，异常类型由 API 层映射（本模块不自带状态码）
    if version is None:
        raise ReviewDetailNotFound("文档不存在")

    # 元数据来源：documents（标题/URL）+ law_versions→laws（类型/机关/日期）；
    # 法规三层表可能未写入（包内无原始 HTML 时元数据会被跳过），故逐级判空
    document = session.get(Document, version.document_id)
    law_version = session.scalar(
        select(LawVersion).where(LawVersion.document_version_id == version.id)
    )
    law = session.get(Law, law_version.law_id) if law_version else None

    # 分块预览：按 sequence 正序（还原条文顺序），只取前 preview_chunks 条
    chunks = session.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_version_id == version.id)
        .order_by(DocumentChunk.sequence.asc())
        .limit(preview_chunks)
    ).all()

    def _truncate(text: str) -> str:
        # 超长截断加省略号：让调用方一眼看出"这里被截了"，不误当成原文结尾
        return text if len(text) <= content_limit else text[:content_limit] + "……"

    return {
        # 文档与版本标识（version_key 是审核动作的入参锚点）
        "document_id": version.document_id,
        "version_key": version.version_key,
        "version_status": version.version_status,
        # 法规元数据（联表拿不到时为 None，前端按"待补录"展示）
        "title": document.title if document else None,
        "document_type": law.document_type if law else None,
        "issuing_authority": law.issuing_authority if law else None,
        "promulgation_date": law_version.promulgation_date.isoformat()
        if law_version and law_version.promulgation_date
        else None,
        "effective_date": law_version.effective_date.isoformat()
        if law_version and law_version.effective_date
        else None,
        "source_url": document.source_url if document else None,
        # 正文指纹：与导入审计对账时用来确认"看的是哪一版内容"
        "content_hash": version.content_hash,
        "created_at": version.created_at.isoformat(),
        # 审核留痕（未审核时三者均 None）
        "reviewed_by": version.reviewed_by,
        "reviewed_at": version.reviewed_at.isoformat() if version.reviewed_at else None,
        "review_note": version.review_note,
        # 分块预览：chunk_key 供前端展开原文，content 已截断
        "chunk_preview": [
            {
                "chunk_key": chunk.chunk_key,
                "article_number": chunk.article_number,
                "content": _truncate(chunk.content),
            }
            for chunk in chunks
        ],
    }
