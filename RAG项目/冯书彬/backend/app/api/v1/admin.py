from fastapi import APIRouter, Depends

from backend.app.api.deps import require_role

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/role-check")
def role_check(_: dict = Depends(require_role("super_admin"))) -> dict[str, bool]:
    # 最小真实后台受保护路由，仅用于验证项目路由已接入角色鉴权。
    return {"ok": True}
