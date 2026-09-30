"""用户接口：资料、偏好角色、修改密码、登录日志。

与其它路由一样，这里只做"取参数 -> 调服务 -> 包装返回"。
所有 /me 前缀的接口都以 get_current_user 作为依赖，天然限定"只能操作自己"，
因为用户 ID 来自登录态（Token），而非客户端传入，从而防止越权访问他人数据。
"""
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from src.api.deps import get_client_ip, get_current_user, revoke_current_token
from src.core.exceptions import ok
from src.db.mysql import get_db
from src.models import User
from src.schemas import PasswordChangeRequest, PreferenceRequest, UserUpdateRequest
from src.services import persona_service, user_service

router = APIRouter(prefix="/users", tags=["用户"])


@router.get("/me", summary="当前用户资料")
def get_me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # user.id 来自 Token 解析，接口路径上没有任何可被猜测的 ID，天然安全。
    return ok(user_service.get_profile(db, user.id))


@router.put("/me", summary="修改当前用户资料")
def update_me(payload: UserUpdateRequest, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    # exclude_none=True 实现"部分更新"：只更新用户提交的字段，避免把未填字段覆盖成空。
    data = user_service.update_profile(db, user.id, payload.model_dump(exclude_none=True))
    return ok(data, "更新成功")


@router.post("/me/password", summary="修改密码（成功后吊销当前 Token）")
def change_password(payload: PasswordChangeRequest, request: Request,
                    user: User = Depends(get_current_user),
                    revoked: bool = Depends(revoke_current_token),
                    db: Session = Depends(get_db)):
    # 修改密码后必须让旧 Token 失效，否则旧凭证仍可继续访问，存在安全风险；
    # revoke_current_token 依赖会在进入函数前先完成吊销，并返回是否成功。
    user_service.change_password(db, user.id, payload.old_password, payload.new_password)
    user_service.add_audit_log(db, user.id, "change_password", ip=get_client_ip(request))
    return ok({"token_revoked": revoked}, "密码修改成功，请重新登录")


@router.get("/me/preferences", summary="我的心理医生偏好")
def list_preferences(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return ok(persona_service.list_user_preferences(db, user.id))


@router.post("/me/preferences", summary="设置默认心理医生（角色切换）")
def set_preference(payload: PreferenceRequest, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    persona_service.set_default_persona(db, user.id, payload.persona_id)
    return ok({"default_persona_id": payload.persona_id}, "已保存默认心理医生")


@router.get("/me/login-logs", summary="我的登录日志")
def my_login_logs(limit: int = Query(20, ge=1, le=200),
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Query(20, ge=1, le=200) 声明了查询参数的默认值与取值范围，
    # FastAPI 会据此自动完成参数校验和 OpenAPI 文档生成。
    return ok(user_service.list_login_logs(db, user.id, limit))