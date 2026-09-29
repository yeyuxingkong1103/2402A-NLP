from collections.abc import Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import require_role
from backend.app.schemas.knowledge_base import (
    CrawlSnapshotCreate,
    CrawlSnapshotRead,
    DeprecateRequest,
    KnowledgeMaterialRead,
    KnowledgeUploadRead,
    KnowledgeUploadRequest,
    ReviewRequest,
    SourceWhitelistCreate,
    SourceWhitelistDisable,
    SourceWhitelistRead,
    SourceWhitelistUpdate,
)
from backend.app.services.knowledge_base_service import (
    KnowledgeBaseWorkflowError,
    create_crawl_snapshot,
    create_source_whitelist_entry,
    delete_source_whitelist_entry,
    deprecate_material,
    disable_source_whitelist_entry,
    list_knowledge_materials,
    publish_material,
    review_material,
    submit_for_review,
    update_source_whitelist_entry,
)

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


def _actor_id(principal: dict) -> str:
    # JWT sub 是内部用户 ID，缺失时使用 unknown 兜底但不记录敏感信息。
    return str(principal.get("sub") or "unknown")


def _map_workflow_error(action: Callable[[], object]) -> object:
    # 状态机错误映射为 409，普通输入错误映射为 400。
    try:
        return action()
    except KnowledgeBaseWorkflowError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/uploads", response_model=KnowledgeUploadRead)
def upload_material(
    body: KnowledgeUploadRequest,
    principal: dict = Depends(require_role("super_admin")),
):
    # 本地上传复用治理流程：授权来源、创建快照、提交审核，避免绕过知识库状态机。
    actor_id = _actor_id(principal)
    source_url = f"local-upload://{uuid4()}-{body.title.strip()}"

    def action() -> dict:
        whitelist = create_source_whitelist_entry(
            source_url,
            body.publisher,
            body.material_type,
            actor_id=actor_id,
            active=True,
        )
        snapshot = create_crawl_snapshot(
            source_url,
            f"<article>{body.content}</article>",
            [{"name": body.title, "url": source_url}],
            actor_id=actor_id,
        )
        material = submit_for_review(snapshot.id, actor_id=actor_id)
        return {"whitelist_id": whitelist.id, "snapshot": snapshot, "material": material}

    return _map_workflow_error(action)


@router.get("/materials", response_model=list[KnowledgeMaterialRead])
def list_materials(
    principal: dict = Depends(require_role("content_reviewer", "super_admin")),
):
    # 审核员和超级管理员可查看治理队列，正文仍由后端响应模型控制。
    return list_knowledge_materials()

@router.post("/source-whitelist", response_model=SourceWhitelistRead)
def create_whitelist(
    body: SourceWhitelistCreate,
    principal: dict = Depends(require_role("super_admin")),
):
    # 只有超级管理员可以维护来源白名单。
    return _map_workflow_error(
        lambda: create_source_whitelist_entry(
            body.url,
            body.publisher,
            body.material_type,
            actor_id=_actor_id(principal),
            active=body.active,
        )
    )


@router.patch("/source-whitelist/{entry_id}", response_model=SourceWhitelistRead)
def update_whitelist(
    entry_id: str,
    body: SourceWhitelistUpdate,
    principal: dict = Depends(require_role("super_admin")),
):
    # 白名单更新仅允许超级管理员修改安全元数据。
    return _map_workflow_error(
        lambda: update_source_whitelist_entry(
            entry_id,
            actor_id=_actor_id(principal),
            url=body.url,
            publisher=body.publisher,
            material_type=body.material_type,
            active=body.active,
        )
    )


@router.post("/source-whitelist/{entry_id}/disable", response_model=SourceWhitelistRead)
def disable_whitelist(
    entry_id: str,
    body: SourceWhitelistDisable,
    principal: dict = Depends(require_role("super_admin")),
):
    # 停用白名单保留记录，供审计和排查使用。
    return _map_workflow_error(lambda: disable_source_whitelist_entry(entry_id, _actor_id(principal), body.reason))


@router.delete("/source-whitelist/{entry_id}", response_model=SourceWhitelistRead)
def delete_whitelist(
    entry_id: str,
    principal: dict = Depends(require_role("super_admin")),
):
    # 删除白名单只撤销授权，不影响既有材料状态。
    return _map_workflow_error(lambda: delete_source_whitelist_entry(entry_id, _actor_id(principal)))


@router.post("/crawl-snapshots", response_model=CrawlSnapshotRead)
def create_snapshot(
    body: CrawlSnapshotCreate,
    principal: dict = Depends(require_role("super_admin")),
):
    # MVP 中由超级管理员触发快照入库；快照不会直接 searchable。
    return _map_workflow_error(
        lambda: create_crawl_snapshot(
            body.source_url,
            body.raw_html,
            body.attachments,
            body.failure_reason,
            body.backup_source_urls,
            actor_id=_actor_id(principal),
        )
    )


@router.post("/crawl-snapshots/{snapshot_id}/submit-review", response_model=KnowledgeMaterialRead)
def submit_snapshot_review(
    snapshot_id: str,
    principal: dict = Depends(require_role("super_admin")),
):
    # 提交审核只生成 pending_review 材料，不发布。
    return _map_workflow_error(lambda: submit_for_review(snapshot_id, actor_id=_actor_id(principal)))


@router.post("/materials/{material_id}/review", response_model=KnowledgeMaterialRead)
def review_knowledge_material(
    material_id: str,
    body: ReviewRequest,
    principal: dict = Depends(require_role("content_reviewer")),
):
    # 内容审核员只能审核通过或拒绝，不能最终发布。
    return _map_workflow_error(lambda: review_material(material_id, _actor_id(principal), body.decision, body.reason))


@router.post("/materials/{material_id}/publish", response_model=KnowledgeMaterialRead)
def publish_knowledge_material(
    material_id: str,
    principal: dict = Depends(require_role("super_admin")),
):
    # 只有超级管理员可以发布 reviewed 材料。
    return _map_workflow_error(lambda: publish_material(material_id, _actor_id(principal)))


@router.post("/materials/{material_id}/deprecate", response_model=KnowledgeMaterialRead)
def deprecate_knowledge_material(
    material_id: str,
    body: DeprecateRequest,
    principal: dict = Depends(require_role("super_admin")),
):
    # 只有超级管理员可以废止已发布材料。
    return _map_workflow_error(lambda: deprecate_material(material_id, _actor_id(principal), body.reason))
