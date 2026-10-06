from typing import Any

from pydantic import BaseModel, Field

from backend.app.models.knowledge_base import MATERIAL_TYPES


class KnowledgeUploadRequest(BaseModel):
    # 前端本地上传先由浏览器读取文本内容，再进入现有白名单、快照和审核流程。
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    publisher: str = Field(default="本地上传", min_length=1)
    material_type: str = "law"



class SourceWhitelistCreate(BaseModel):
    # 白名单创建请求只允许声明来源、发布机关和材料类型。
    url: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    material_type: str
    active: bool = True


class SourceWhitelistUpdate(BaseModel):
    # 白名单更新允许最小字段变更，不接收任何正文。
    url: str | None = Field(default=None, min_length=1)
    publisher: str | None = Field(default=None, min_length=1)
    material_type: str | None = None
    active: bool | None = None


class SourceWhitelistDisable(BaseModel):
    # 停用原因只用于审计，不影响已保存快照。
    reason: str = Field(min_length=1)


class SourceWhitelistRead(BaseModel):
    # API 返回白名单状态，不包含任何抓取正文。
    id: str
    url: str
    publisher: str
    material_type: str
    active: bool


class CrawlSnapshotCreate(BaseModel):
    # 抓取快照接收原始 HTML，但日志和审计不得记录该字段明文。
    source_url: str = Field(min_length=1)
    raw_html: str = Field(min_length=1)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    backup_source_urls: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class CrawlSnapshotRead(BaseModel):
    # 快照响应只暴露留痕摘要和状态，不返回 raw_text。
    id: str
    source_url: str
    publisher: str
    material_type: str
    raw_content_hash: str
    attachments: list[dict[str, Any]]
    backup_source_urls: list[str]
    status: str
    searchable: bool
    failure_reason: str | None = None


class ReviewRequest(BaseModel):
    # decision 仅允许 approved 或 rejected，由服务层映射到状态。
    decision: str
    reason: str = Field(min_length=1)


class DeprecateRequest(BaseModel):
    # 废止必须填写原因，便于审计追踪。
    reason: str = Field(min_length=1)


class KnowledgeMaterialRead(BaseModel):
    # 材料响应用于治理流程，不提供检索结果或 RAG 字段。
    id: str
    snapshot_id: str
    source_url: str
    publisher: str
    material_type: str
    status: str
    searchable: bool
    reviewed_by: str | None = None
    published_by: str | None = None
    effective_from: str | None = None


class KnowledgeUploadRead(BaseModel):
    # 上传响应串起来源白名单、快照和待审核材料，方便前端展示完整链路。
    whitelist_id: str
    snapshot: CrawlSnapshotRead
    material: KnowledgeMaterialRead


def validate_material_type(material_type: str) -> str:
    # Pydantic v1/v2 兼容地把枚举校验集中到服务入口复用。
    if material_type not in MATERIAL_TYPES:
        raise ValueError("invalid material_type")
    return material_type
