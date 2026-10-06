from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


MATERIAL_TYPES = {
    "law",
    "administrative_regulation",
    "department_rule",
    "judicial_interpretation",
    "guiding_case",
    "typical_case",
}

MATERIAL_STATUSES = {"crawled", "pending_review", "reviewed", "rejected", "published", "deprecated"}


@dataclass
class SourceWhitelistEntry:
    # 来源白名单只保存来源域和发文机关，不保存抓取正文。
    id: str
    url: str
    publisher: str
    material_type: str
    active: bool
    created_by: str
    created_at: datetime
    updated_at: datetime


@dataclass
class CrawlSnapshot:
    # 抓取快照是原始留痕，不是正式知识库内容。
    id: str
    source_url: str
    publisher: str
    material_type: str
    raw_content_hash: str
    raw_text: str
    attachments: list[dict[str, Any]]
    backup_source_urls: list[str]
    crawled_at: datetime
    status: str = "crawled"
    searchable: bool = False
    failure_reason: str | None = None


@dataclass
class MaterialAttachment:
    # 附件元数据从快照复制到材料，便于审核时查看来源文件。
    name: str
    url: str
    content_hash: str | None = None


@dataclass
class ReviewRecord:
    # 审核记录追加保存审核人、结论和原因。
    reviewer_id: str
    decision: str
    reason: str
    reviewed_at: datetime


@dataclass
class PublishRecord:
    # 发布记录保存最终发布人和时间，发布后才可检索。
    publisher_id: str
    published_at: datetime


@dataclass
class StatusHistory:
    # 状态历史记录每次状态迁移的操作者和原因。
    from_status: str | None
    to_status: str
    actor_id: str
    reason: str
    changed_at: datetime


@dataclass
class KnowledgeMaterial:
    # 正式材料从快照提交审核后生成，默认不可检索。
    id: str
    snapshot_id: str
    source_url: str
    publisher: str
    material_type: str
    raw_text: str
    attachments: list[MaterialAttachment]
    status: str
    searchable: bool
    created_at: datetime
    updated_at: datetime
    reviewed_by: str | None = None
    published_by: str | None = None
    effective_from: str | None = None
    review_records: list[ReviewRecord] = field(default_factory=list)
    publish_records: list[PublishRecord] = field(default_factory=list)
    status_history: list[StatusHistory] = field(default_factory=list)
