"""replace_existing 模式下的旧版本清理。

原 import_service.py 并入部分：删除指定文档的所有旧版本及其关联记录，
按外键依赖顺序逐层删除（chunk → import_record → crawl_record →
articles → law_versions → document_versions），并重置文档当前版本指针。
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.sql_models import (
    Article,
    CrawlRecord,
    Document,
    DocumentChunk,
    DocumentVersion,
    ImportRecord,
    LawVersion,
)


def _delete_existing_versions_and_chunks(session: Session, document_id: int) -> None:
    """删除指定文档的所有旧版本及其关联记录（事务内执行，失败回滚）。"""
    from sqlalchemy import delete

    # 先查出该文档的所有版本 ID
    version_ids = [
        row[0] for row in session.execute(
            select(DocumentVersion.id).where(DocumentVersion.document_id == document_id)
        ).fetchall()
    ]

    if not version_ids:
        return

    # 按依赖顺序删除（先删除引用方，再删除被引用方）
    # 1. 删除所有版本关联的 chunk
    for version_id in version_ids:
        session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_version_id == version_id)
        )

    # 2. 删除 ImportRecord（引用 document_version_id）
    for version_id in version_ids:
        session.execute(
            delete(ImportRecord).where(ImportRecord.document_version_id == version_id)
        )

    # 3. 删除 CrawlRecord（引用 document_version_id）
    session.execute(
        delete(CrawlRecord).where(CrawlRecord.document_id == document_id)
    )

    # 4. 删除 law_versions 和 articles（先查出关联的 law_version_ids）
    law_version_ids = [
        row[0] for row in session.execute(
            select(LawVersion.id).where(LawVersion.document_version_id.in_(version_ids))
        ).fetchall()
    ]
    if law_version_ids:
        # 先删除 articles（引用 law_version_id）
        session.execute(
            delete(Article).where(Article.law_version_id.in_(law_version_ids))
        )
        # 再删除 law_versions
        session.execute(
            delete(LawVersion).where(LawVersion.id.in_(law_version_ids))
        )

    # 5. 重置文档的 current_version_id（避免外键约束冲突）
    document = session.get(Document, document_id)
    if document:
        document.current_version_id = None
    session.flush()

    # 6. 删除所有版本记录
    session.execute(
        delete(DocumentVersion).where(DocumentVersion.document_id == document_id)
    )
