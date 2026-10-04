"""角色接口：列表、详情、创建。角色列表/详情/创建，直接调用"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.mysql_store import create_role as db_create_role
from app.db.mysql_store import get_role as db_get_role
from app.db.mysql_store import list_roles as db_list_roles
from app.logging_conf import log
from app.schemas import RoleCreate

router = APIRouter(tags=["roles"])


@router.get("/roles")
def list_roles() -> list[dict]:
    return db_list_roles()


@router.get("/roles/{role_id}")
def get_role(role_id: int) -> dict:
    role = db_get_role(role_id)
    if not role:
        raise HTTPException(404, "角色不存在")
    return role


@router.post("/roles")
def create_role(payload: RoleCreate) -> dict:
    try:
        new_id = db_create_role(
            payload.role_name, payload.domain, payload.description,
            payload.opening, payload.system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("创建角色失败: %s", exc)
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "id": new_id}
