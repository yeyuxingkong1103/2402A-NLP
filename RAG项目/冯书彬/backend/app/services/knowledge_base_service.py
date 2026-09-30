from datetime import timezone
from hashlib import sha256
from html import unescape
from re import sub
from uuid import uuid4

from backend.app.core.security import utc_now
from backend.app.models.knowledge_base import (
    MATERIAL_TYPES,
    CrawlSnapshot,
    KnowledgeMaterial,
    MaterialAttachment,
    PublishRecord,
    ReviewRecord,
    SourceWhitelistEntry,
    StatusHistory,
)
from backend.app.repositories.knowledge_base_repository import (
    delete_whitelist_entry,
    find_active_whitelist_for_url,
    find_whitelist_by_url,
    get_material,
    get_snapshot,
    get_whitelist_entry,
    list_materials,
    save_material,
    save_snapshot,
    save_whitelist_entry,
)
from backend.app.services.audit_service import AuditAction, record_audit


class KnowledgeBaseWorkflowError(ValueError):
    # 业务状态错误使用 ValueError 子类，便于 API 层映射 409。
    pass


def _now():
    # 统一使用 UTC 时间，避免本地时区影响审计和状态历史。
    return utc_now().astimezone(timezone.utc)


def _require_material_type(material_type: str) -> None:
    # 材料类型必须限定在法律治理白名单内。
    if material_type not in MATERIAL_TYPES:
        raise ValueError("invalid material_type")


def _extract_text(raw_html: str) -> str:
    # 最小 HTML 文本提取，仅用于保存快照和材料正文，不做 RAG 清洗。
    text = sub(r"<[^>]+>", " ", raw_html)
    # 合并空白并反转义 HTML 实体，避免保留标签。
    return " ".join(unescape(text).split())


def _hash_content(raw_html: str) -> str:
    # 原文摘要用于留痕去重，不在审计中记录原文。
    return sha256(raw_html.encode("utf-8")).hexdigest()


def _history(from_status: str | None, to_status: str, actor_id: str, reason: str) -> StatusHistory:
    # 状态历史统一构造，确保每次迁移都有原因。
    return StatusHistory(from_status=from_status, to_status=to_status, actor_id=actor_id, reason=reason, changed_at=_now())


def _whitelist_metadata(entry: SourceWhitelistEntry, operation: str) -> dict[str, object]:
    # 白名单审计只记录安全字段，不记录抓取正文或页面内容。
    return {"operation": operation, "entry_id": entry.id, "url": entry.url, "publisher": entry.publisher, "material_type": entry.material_type, "active": entry.active}


def _snapshot_metadata(snapshot: CrawlSnapshot, operation: str, actor_id: str) -> dict[str, object]:
    # 快照审计记录摘要和状态，禁止记录 raw_text/raw_html。
    return {
        "operation": operation,
        "snapshot_id": snapshot.id,
        "status": snapshot.status,
        "actor_id": actor_id,
        "source_url": snapshot.source_url,
        "publisher": snapshot.publisher,
        "material_type": snapshot.material_type,
        "raw_content_hash": snapshot.raw_content_hash,
        "attachment_count": len(snapshot.attachments),
        "backup_source_count": len(snapshot.backup_source_urls),
        "failure_reason": snapshot.failure_reason,
    }


def _material_metadata(material: KnowledgeMaterial, operation: str, actor_id: str, reason: str | None = None) -> dict[str, object]:
    # 材料审计记录治理状态和操作者，不记录正文原文。
    metadata: dict[str, object] = {
        "operation": operation,
        "material_id": material.id,
        "snapshot_id": material.snapshot_id,
        "status": material.status,
        "actor_id": actor_id,
        "source_url": material.source_url,
        "publisher": material.publisher,
        "material_type": material.material_type,
        "searchable": material.searchable,
    }
    if reason is not None:
        metadata["reason"] = reason
    return metadata


def _validate_backup_sources(source_entry: SourceWhitelistEntry, backup_source_urls: list[str]) -> list[str]:
    # 备用来源必须命中启用白名单，且发布机关必须与主来源一致。
    normalized_urls = [url.rstrip("/") for url in backup_source_urls]
    for backup_url in normalized_urls:
        backup_entry = find_active_whitelist_for_url(backup_url)
        if not backup_entry:
            raise ValueError("active whitelist entry is required for backup source")
        if backup_entry.publisher != source_entry.publisher:
            raise ValueError("backup source publisher must match primary publisher")
    return normalized_urls


def create_source_whitelist_entry(
    url: str,
    publisher: str,
    material_type: str,
    actor_id: str = "system",
    active: bool = True,
) -> SourceWhitelistEntry:
    # 只有 API 层 super_admin 可调用；服务层负责数据合法性和审计。
    _require_material_type(material_type)
    if find_whitelist_by_url(url):
        raise ValueError("source whitelist entry already exists")
    entry = SourceWhitelistEntry(
        id=str(uuid4()),
        url=url.rstrip("/"),
        publisher=publisher,
        material_type=material_type,
        active=active,
        created_by=actor_id,
        created_at=_now(),
        updated_at=_now(),
    )
    save_whitelist_entry(entry)
    record_audit(AuditAction.WHITELIST_CHANGE, actor_id, "source_whitelist", entry.id, _whitelist_metadata(entry, "create"))
    return entry


def update_source_whitelist_entry(
    entry_id: str,
    actor_id: str,
    url: str | None = None,
    publisher: str | None = None,
    material_type: str | None = None,
    active: bool | None = None,
) -> SourceWhitelistEntry:
    entry = get_whitelist_entry(entry_id)
    if not entry:
        raise ValueError("source whitelist entry not found")
    if material_type is not None:
        _require_material_type(material_type)
    normalized_url = url.rstrip("/") if url is not None else None
    duplicate = find_whitelist_by_url(normalized_url) if normalized_url else None
    if duplicate and duplicate.id != entry_id:
        raise ValueError("source whitelist entry already exists")
    if normalized_url is not None:
        entry.url = normalized_url
    if publisher is not None:
        entry.publisher = publisher
    if material_type is not None:
        entry.material_type = material_type
    if active is not None:
        entry.active = active
    entry.updated_at = _now()
    save_whitelist_entry(entry)
    record_audit(AuditAction.WHITELIST_CHANGE, actor_id, "source_whitelist", entry.id, _whitelist_metadata(entry, "update"))
    return entry


def disable_source_whitelist_entry(entry_id: str, actor_id: str, reason: str) -> SourceWhitelistEntry:
    # 停用是软删除路径，保留来源记录用于后续审计追踪。
    entry = update_source_whitelist_entry(entry_id, actor_id=actor_id, active=False)
    record_audit(AuditAction.WHITELIST_CHANGE, actor_id, "source_whitelist", entry.id, {**_whitelist_metadata(entry, "disable"), "reason": reason})
    return entry


def delete_source_whitelist_entry(entry_id: str, actor_id: str) -> SourceWhitelistEntry:
    # 删除仅撤销白名单授权，不删除快照、材料和审计记录。
    entry = delete_whitelist_entry(entry_id)
    if not entry:
        raise ValueError("source whitelist entry not found")
    entry.updated_at = _now()
    record_audit(AuditAction.WHITELIST_CHANGE, actor_id, "source_whitelist", entry.id, _whitelist_metadata(entry, "delete"))
    return entry


def create_crawl_snapshot(
    source_url: str,
    raw_html: str,
    attachments: list[dict],
    failure_reason: str | None = None,
    backup_source_urls: list[str] | None = None,
    actor_id: str = "system",
) -> CrawlSnapshot:
    # 抓取前必须命中启用白名单，避免非授权来源进入治理流程。
    entry = find_active_whitelist_for_url(source_url.rstrip("/"))
    if not entry:
        raise ValueError("active whitelist entry is required")
    normalized_backups = _validate_backup_sources(entry, backup_source_urls or [])
    snapshot = CrawlSnapshot(
        id=str(uuid4()),
        source_url=source_url.rstrip("/"),
        publisher=entry.publisher,
        material_type=entry.material_type,
        raw_content_hash=_hash_content(raw_html),
        raw_text=_extract_text(raw_html),
        attachments=attachments,
        backup_source_urls=normalized_backups,
        crawled_at=_now(),
        status="crawled",
        searchable=False,
        failure_reason=failure_reason,
    )
    saved = save_snapshot(snapshot)
    record_audit(AuditAction.KB_SNAPSHOT_CREATE, actor_id, "crawl_snapshot", saved.id, _snapshot_metadata(saved, "create_snapshot", actor_id))
    return saved


def submit_for_review(snapshot_id: str, actor_id: str = "system") -> KnowledgeMaterial:
    # 只有 crawled 快照可以提交审核，快照本身不变成可检索内容。
    snapshot = get_snapshot(snapshot_id)
    if not snapshot:
        raise ValueError("snapshot not found")
    if snapshot.status != "crawled":
        raise KnowledgeBaseWorkflowError("snapshot must be crawled before review")
    snapshot.status = "pending_review"
    material = KnowledgeMaterial(
        id=str(uuid4()),
        snapshot_id=snapshot.id,
        source_url=snapshot.source_url,
        publisher=snapshot.publisher,
        material_type=snapshot.material_type,
        raw_text=snapshot.raw_text,
        attachments=[MaterialAttachment(**attachment) for attachment in snapshot.attachments],
        status="pending_review",
        searchable=False,
        created_at=_now(),
        updated_at=_now(),
    )
    material.status_history.append(_history("crawled", "pending_review", actor_id, "submitted for review"))
    saved = save_material(material)
    record_audit(AuditAction.KB_SUBMIT_REVIEW, actor_id, "knowledge_material", saved.id, _material_metadata(saved, "submit_review", actor_id))
    return saved


def review_material(material_id: str, reviewer_id: str, decision: str, reason: str) -> KnowledgeMaterial:
    # 内容审核员只能将 pending_review 改为 reviewed 或 rejected，不能发布。
    material = get_material(material_id)
    if not material:
        raise ValueError("material not found")
    if material.status != "pending_review":
        raise KnowledgeBaseWorkflowError("material must be pending_review before review")
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    next_status = "reviewed" if decision == "approved" else "rejected"
    material.status = next_status
    material.searchable = False
    material.reviewed_by = reviewer_id
    material.updated_at = _now()
    material.review_records.append(ReviewRecord(reviewer_id, decision, reason, _now()))
    material.status_history.append(_history("pending_review", next_status, reviewer_id, reason))
    action = AuditAction.KB_REVIEW if next_status == "reviewed" else AuditAction.KB_REJECT
    record_audit(action, reviewer_id, "knowledge_material", material.id, _material_metadata(material, next_status, reviewer_id, reason))
    return save_material(material)


def publish_material(material_id: str, publisher_id: str) -> KnowledgeMaterial:
    # 只有 reviewed 材料可发布，发布后才 searchable。
    material = get_material(material_id)
    if not material:
        raise ValueError("material not found")
    if material.status != "reviewed":
        raise KnowledgeBaseWorkflowError("material must be reviewed before publish")
    material.status = "published"
    material.searchable = True
    material.published_by = publisher_id
    material.updated_at = _now()
    material.publish_records.append(PublishRecord(publisher_id, _now()))
    material.status_history.append(_history("reviewed", "published", publisher_id, "published"))
    record_audit(AuditAction.KB_PUBLISH, publisher_id, "knowledge_material", material.id, _material_metadata(material, "publish", publisher_id))
    return save_material(material)


def list_knowledge_materials() -> list[KnowledgeMaterial]:
    # 审核页面只读取材料状态和治理元数据，不返回不必要的正文。
    return list_materials()


def deprecate_material(material_id: str, actor_id: str, reason: str) -> KnowledgeMaterial:
    # 只有 published 材料可废止，废止后立刻不可检索。
    material = get_material(material_id)
    if not material:
        raise ValueError("material not found")
    if material.status != "published":
        raise KnowledgeBaseWorkflowError("material must be published before deprecate")
    material.status = "deprecated"
    material.searchable = False
    material.updated_at = _now()
    material.status_history.append(_history("published", "deprecated", actor_id, reason))
    record_audit(AuditAction.KB_DEPRECATE, actor_id, "knowledge_material", material.id, _material_metadata(material, "deprecate", actor_id, reason))
    return save_material(material)
