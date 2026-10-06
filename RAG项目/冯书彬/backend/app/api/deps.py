import logging
from collections.abc import Callable
from datetime import datetime, timezone

from fastapi import Header, HTTPException

from backend.app.core.config import settings
from backend.app.core.security import decode_access_token
from backend.app.models.role import is_known_role
from backend.app.models.user import User
from backend.app.services.auth_service import get_auth_store

logger = logging.getLogger(__name__)


def _ensure_dev_user(user_id: str) -> None:
    # 开发免登录会绕过真实登录流程；SQL 后端仍需要用户记录满足外键约束。
    store = get_auth_store()
    if store.get_user_by_id(user_id) is not None:
        return
    now = datetime.now(timezone.utc)
    store.save_user(
        User(
            id=user_id,
            encrypted_phone="dev-auth-bypass-placeholder",
            phone_nonce="dev-auth-bypass-placeholder",
            phone_encrypted_data_key="dev-auth-bypass-placeholder",
            phone_key_version="dev",
            phone_hmac=f"dev-auth-bypass:{user_id}",
            agreement_version="dev",
            privacy_policy_version="dev",
            adult_confirmed=True,
            created_at=now,
            updated_at=now,
        )
    )
    logger.info("development auth user created", extra={"user_id": user_id})


def _dev_principal(user_id: str | None) -> dict | None:
    # 本地联调允许前端显式携带演示身份头；生产环境必须关闭。
    if settings.ENVIRONMENT not in {"development", "test", "testing"} or not settings.DEV_AUTH_BYPASS or not user_id:
        return None
    _ensure_dev_user(user_id)
    roles = [role.strip() for role in settings.DEV_AUTH_ROLES.split(",") if role.strip()]
    return {"sub": user_id, "sid": "dev-browser-session", "roles": roles}


def _extract_bearer_token(authorization: str | None) -> str:
    # Authorization 缺失或格式错误统一按未认证处理。
    if not authorization:
        raise HTTPException(status_code=401, detail="missing bearer token")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="invalid bearer token")
    return token


def _reject_deleted_user(payload: dict) -> None:
    # 注销用户的旧 access token 仍可能未过期，必须在鉴权入口统一拒绝。
    user_id = payload.get("sub")
    if not user_id or get_auth_store().is_user_deleted(user_id):
        logger.warning("access token rejected for deleted user", extra={"user_id": user_id})
        raise HTTPException(status_code=401, detail="invalid access token")


def get_current_principal(authorization: str | None = Header(default=None), dev_user_id: str | None = Header(default=None, alias="X-Dev-User-Id")) -> dict:
    # 只解码访问令牌，不记录 token 明文；本地联调可使用演示身份头。
    dev_principal = _dev_principal(dev_user_id)
    if dev_principal and not authorization:
        return dev_principal
    token = _extract_bearer_token(authorization)
    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        logger.warning("access token rejected")
        raise HTTPException(status_code=401, detail="invalid access token") from exc
    _reject_deleted_user(payload)
    return payload


def require_role(*roles: str) -> Callable[[str | None], dict]:
    # 依赖创建时先校验角色名，避免路由配置拼写错误。
    unknown_roles = [role for role in roles if not is_known_role(role)]
    if unknown_roles:
        raise ValueError(f"unknown roles: {', '.join(unknown_roles)}")
    allowed_roles = set(roles)

    def dependency(
        authorization: str | None = Header(default=None),
        dev_user_id: str | None = Header(default=None, alias="X-Dev-User-Id"),
    ) -> dict:
        # 每次请求解析当前用户并检查 JWT roles 声明。
        principal = get_current_principal(authorization, dev_user_id)
        principal_roles = set(principal.get("roles") or [])
        if principal_roles.isdisjoint(allowed_roles):
            logger.warning(
                "role permission denied",
                extra={"user_id": principal.get("sub"), "required_roles": sorted(allowed_roles)},
            )
            raise HTTPException(status_code=403, detail="insufficient role")
        return principal

    return dependency
