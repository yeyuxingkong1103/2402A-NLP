import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.import_cleanup import _delete_existing_versions_and_chunks
from app.db.import_version_status import decide_version_status
from app.db.legal_metadata_writer import _add_legal_metadata
from app.db.sql_models import (
    CrawlRecord,
    Document,
    DocumentChunk,
    DocumentVersion,
    ImportRecord,
    utc_now,
)
from app.pipeline.package_validator import ValidatedPackage
from app.db.version_status import PENDING_REVIEW


@dataclass(frozen=True)
class ImportPackageResult:
    """数据包入库结果；CLI 只输出这些脱敏字段。"""

    status: str
    package_id: str
    version_status: str | None = None
    document_key: str | None = None
    version_key: str | None = None
    error_summary: str | None = None
    # updated 的细分原因：chunking_changed = 正文未变但切块方式变化；
    # 正文变化或首次导入时为 None
    reason: str | None = None


def import_package(session: Session, package: ValidatedPackage, replace_existing: bool = False) -> ImportPackageResult:
    """把已校验数据包事务导入 MySQL 权威存储。"""
    package_id = package.manifest.package_id
    existing_import = _get_import_record(session, package_id)
    if not replace_existing and existing_import is not None and existing_import.status in {
        "imported",
        "already_imported",
    }:
        return ImportPackageResult(
            status="already_imported",
            package_id=package_id,
            version_status="unchanged",
            document_key=package.data.document.document_id,
            version_key=package.data.version.version_id,
        )

    try:
        result = _import_package_rows(session, package, replace_existing)
        session.commit()
        return result
    except Exception as error:
        session.rollback()
        error_summary = _safe_error_summary(error)
        _record_failed_import(session, package, error_summary)
        session.commit()
        return ImportPackageResult(
            status="failed",
            package_id=package_id,
            error_summary=error_summary,
        )


def _import_package_rows(session: Session, package: ValidatedPackage, replace_existing: bool = False) -> ImportPackageResult:
    """执行实际的数据库写入操作（在事务内）。

    增量判定逻辑（双比较：content_hash + chunk_fingerprint）已抽到
    app/db/import_version_status.py 的 decide_version_status，本函数
    只消费判定结果并落库；unchanged 分支的幂等审计仍在这里处理。
    """
    data = package.data
    document = _get_or_create_document(session, package, update_title=replace_existing)

    # 增量判定（双比较），规则与存量兼容语义见 import_version_status.py
    decision = decide_version_status(session, document, data.version)
    if decision.version_status == "unchanged":
        unchanged_version = decision.unchanged_version
        # unchanged 分支不再追加 crawl_records（幂等）：
        # 采集审计记录的是"抓取行为"，包未变化时没有新采集发生，
        # 每次重导都追加会让该表随重复导入无限增长（用户裁决 #6）。
        # import_record 仍按幂等更新（状态 already_imported）。
        logging.getLogger(__name__).info(
            "数据包未变化，跳过 crawl_record 追加：package_id=%s",
            package.manifest.package_id,
        )
        _add_import_record(session, package, "already_imported", unchanged_version.id)
        return ImportPackageResult(
            status="already_imported",
            package_id=package.manifest.package_id,
            version_status="unchanged",
            document_key=data.document.document_id,
            version_key=unchanged_version.version_key,
        )
    version_status = decision.version_status
    chunking_changed = decision.chunking_changed

    # 如果是 updated 且 replace_existing=True，在创建新版本之前删除旧版本
    if version_status == "updated" and replace_existing:
        _delete_existing_versions_and_chunks(session, document.id)

    version = DocumentVersion(
        version_key=data.version.version_id,
        document_id=document.id,
        content_hash=data.version.content_hash,
        # 切块指纹随版本落库；None 允许（兼容无指纹的包）
        chunk_fingerprint=data.version.chunk_fingerprint,
        raw_file_path=data.version.raw_file_path,
        media_type=data.version.media_type,
        cleaned_content=data.version.cleaned_content,
        # 阶段6：库列 version_status 是审核状态，新版本一律待审核；
        # （result 里的 version_status 保留 new/updated/unchanged 导入审计语义）
        version_status=PENDING_REVIEW,
        processing_status="awaiting_embedding",
    )
    session.add(version)
    session.flush()
    _add_chunks(session, package, version.id)

    # 写入三层表（laws / law_versions / articles）。
    # 元数据缺失（原始 HTML 不在磁盘、日期抽不到等）不应让整篇文档导入失败：
    # 正文与分块是核心资产，元数据属于可后续补录的信息，
    # 因此这里捕获并告警，保证文档本体照常入库。
    try:
        _add_legal_metadata(session, package, version.id)
    except (FileNotFoundError, ValueError) as error:
        logging.getLogger(__name__).warning(
            "元数据写入跳过：package_id=%s 原因=%s",
            package.manifest.package_id,
            type(error).__name__,
        )

    _add_crawl_record(session, package, document.id, version.id, "success")
    _add_import_record(session, package, "imported", version.id)
    document.current_version_id = version.id
    return ImportPackageResult(
        status="imported",
        package_id=package.manifest.package_id,
        version_status=version_status,
        document_key=data.document.document_id,
        version_key=data.version.version_id,
        # 切块变化场景标记原因，供 CLI/审计区分"正文修订"与"切块重建"
        reason="chunking_changed" if chunking_changed else None,
    )


def _get_or_create_document(session: Session, package: ValidatedPackage, update_title: bool = False) -> Document:
    entry = package.data.document
    document = session.scalar(select(Document).where(Document.source_url == entry.source_url))
    if document is not None:
        # replace_existing 模式下更新 title（可能从哈希改为可读法规名）
        if update_title:
            document.title = entry.title
        return document
    document = Document(
        document_key=entry.document_id,
        source_url=entry.source_url,
        title=entry.title,
    )
    session.add(document)
    session.flush()
    return document


def _add_chunks(session: Session, package: ValidatedPackage, version_id: int) -> None:
    seen_chunk_keys: set[str] = set()
    for chunk in package.data.chunks:
        if chunk.chunk_id in seen_chunk_keys:
            raise ValueError("duplicated chunk_id")
        seen_chunk_keys.add(chunk.chunk_id)
        session.add(
            DocumentChunk(
                chunk_key=chunk.chunk_id,
                document_version_id=version_id,
                parent_chunk_key=chunk.parent_chunk_id,
                chunk_type=chunk.chunk_type,
                article_number=chunk.article_number,
                paragraph_number=chunk.paragraph_number,
                item_number=chunk.item_number,
                sequence=chunk.sequence,
                content=chunk.content,
                retrieval_text=chunk.retrieval_text,
            )
        )


def _add_crawl_record(
    session: Session,
    package: ValidatedPackage,
    document_id: int,
    version_id: int,
    status: str,
) -> None:
    crawl = package.data.crawl
    session.add(
        CrawlRecord(
            package_id=package.manifest.package_id,
            source_url=crawl.source_url,
            document_id=document_id,
            document_version_id=version_id,
            status=status,
            http_status=crawl.http_status,
            content_hash=crawl.content_hash,
            raw_file_path=crawl.raw_file_path,
        )
    )


def _add_import_record(
    session: Session,
    package: ValidatedPackage,
    status: str,
    version_id: int | None,
) -> None:
    """写入/更新数据包级导入审计记录（幂等）。

    import_records.package_id 有 UNIQUE 约束，重复导入（如 replace_existing
    重跑 unchanged 分支）不得再插新行，否则必撞 1062 重复键。
    幂等写法与 _record_failed_import 保持一致：按 package_id 先查再写——
    已存在则更新 status / document_version_id / updated_at，不存在才插入。
    """
    existing = session.scalar(
        select(ImportRecord).where(ImportRecord.package_id == package.manifest.package_id)
    )
    if existing:
        existing.status = status
        existing.document_version_id = version_id
        existing.updated_at = utc_now()
        return
    session.add(
        ImportRecord(
            package_id=package.manifest.package_id,
            schema_version=package.manifest.schema_version,
            pipeline_version=package.manifest.pipeline_version,
            status=status,
            document_version_id=version_id,
        )
    )


def _record_failed_import(
    session: Session,
    package: ValidatedPackage,
    error_summary: str,
) -> None:
    existing = session.scalar(
        select(ImportRecord).where(ImportRecord.package_id == package.manifest.package_id)
    )
    if existing:
        existing.status = "failed"
        existing.error_summary = error_summary
        existing.updated_at = utc_now()
    else:
        session.add(
            ImportRecord(
                package_id=package.manifest.package_id,
                schema_version=package.manifest.schema_version,
                pipeline_version=package.manifest.pipeline_version,
                status="failed",
                error_summary=error_summary,
            )
        )


def _get_import_record(session: Session, package_id: str) -> ImportRecord | None:
    return session.scalar(select(ImportRecord).where(ImportRecord.package_id == package_id))


def _safe_error_summary(error: Exception) -> str:
    """只保留异常类型，避免把正文或连接串写进审计记录。"""
    if isinstance(error, SQLAlchemyError):
        return error.__class__.__name__[:120]
    return error.__class__.__name__[:120]
