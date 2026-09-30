from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.api.deps import get_current_principal
from backend.app.services.export_service import claim_export_password, download_export_job, export_user_data, issue_second_factor_token
from backend.app.services.memory_service import (
    delete_memory,
    list_user_memories,
    memory_to_export_dict,
    set_memory_auto_extraction,
    update_memory,
)
from backend.app.services.user_service import delete_user_account

router = APIRouter(prefix="/api/v1/users", tags=["users"])


class ExportRequest(BaseModel):
    second_factor_token: str | None = None


class ExportPasswordRequest(BaseModel):
    second_factor_token: str | None = None


class DeleteAccountRequest(BaseModel):
    confirmed: bool = False


class MemoryPreferenceRequest(BaseModel):
    enabled: bool


class MemoryUpdateRequest(BaseModel):
    facts: dict
    summary: str


def _require_same_user(path_user_id: str, principal: dict) -> None:
    # 所有个人数据接口以认证用户为准，拒绝路径 user_id 越权访问。
    if principal.get("sub") != path_user_id:
        raise HTTPException(status_code=403, detail="无权访问该用户数据")


@router.post("/{user_id}/export/challenge")
def create_export_challenge(user_id: str, principal: dict = Depends(get_current_principal)):
    _require_same_user(user_id, principal)
    return {"second_factor_token": issue_second_factor_token(user_id)}


@router.post("/{user_id}/export")
def create_export(user_id: str, body: ExportRequest, principal: dict = Depends(get_current_principal)):
    # API 只返回导出作业元数据，不回传 ZIP 密码或加密文件正文。
    _require_same_user(user_id, principal)
    try:
        job = export_user_data(user_id, second_factor_token=body.second_factor_token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"id": job.id, "expires_at": job.expires_at.isoformat(), "max_downloads": job.max_downloads, "download_count": job.download_count}


@router.get("/{user_id}/export/{job_id}/download")
def download_export(user_id: str, job_id: str, principal: dict = Depends(get_current_principal)):
    _require_same_user(user_id, principal)
    try:
        downloaded = download_export_job(job_id, user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"encrypted_zip": downloaded["encrypted_zip"], "password_delivery": downloaded["password_delivery"]}


@router.post("/{user_id}/export/{job_id}/password")
def get_export_password(user_id: str, job_id: str, body: ExportPasswordRequest, principal: dict = Depends(get_current_principal)):
    # 口令必须经二次验证单独领取，且服务端只返回一次。
    _require_same_user(user_id, principal)
    try:
        password = claim_export_password(job_id, user_id, body.second_factor_token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"zip_password": password}


@router.delete("/{user_id}")
def delete_account(user_id: str, body: DeleteAccountRequest, principal: dict = Depends(get_current_principal)):
    # 注销需要显式确认，避免误删。
    _require_same_user(user_id, principal)
    if not body.confirmed:
        raise HTTPException(status_code=400, detail="注销账号需要确认")
    delete_user_account(user_id)
    return {"deleted": True}


@router.get("/{user_id}/memories")
def get_memories(user_id: str, principal: dict = Depends(get_current_principal)):
    # 返回脱敏导出视图，避免暴露密文字段和内部密钥元数据。
    _require_same_user(user_id, principal)
    return {"memories": [memory_to_export_dict(memory) for memory in list_user_memories(user_id)]}


@router.patch("/{user_id}/memory-preferences")
def set_memory_preferences(user_id: str, body: MemoryPreferenceRequest, principal: dict = Depends(get_current_principal)):
    _require_same_user(user_id, principal)
    enabled = set_memory_auto_extraction(user_id, body.enabled)
    return {"memory_auto_extraction_enabled": enabled}


@router.put("/{user_id}/memories/{memory_id}")
def edit_memory(user_id: str, memory_id: str, body: MemoryUpdateRequest, principal: dict = Depends(get_current_principal)):
    _require_same_user(user_id, principal)
    memory = update_memory(memory_id, user_id, body.facts, body.summary)
    if memory is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return memory_to_export_dict(memory)


@router.delete("/{user_id}/memories/{memory_id}")
def remove_memory(user_id: str, memory_id: str, principal: dict = Depends(get_current_principal)):
    _require_same_user(user_id, principal)
    if not delete_memory(memory_id, user_id):
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"deleted": True}
