from collections.abc import Callable

from fastapi import APIRouter, Header, HTTPException

from backend.app.schemas.auth import LoginBody, LogoutBody, OtpRequestResult, RefreshBody, RequestCodeBody, TokenPair
from backend.app.services.auth_service import (
    AuthServiceError,
    login_with_code,
    refresh_access_token,
    request_login_code,
    revoke_refresh_token,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _map_auth_error(action: Callable[[], TokenPair | OtpRequestResult | None]) -> TokenPair | OtpRequestResult | None:
    # 服务层业务错误统一转成 4xx，避免 FastAPI 返回 500。
    try:
        return action()
    except AuthServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/code", response_model=OtpRequestResult)
def request_code(body: RequestCodeBody) -> OtpRequestResult:
    # 只返回是否模拟发送和有效期，不回显手机号或验证码。
    return _map_auth_error(lambda: request_login_code(body.phone, client_id=body.client_id))


@router.post("/login", response_model=TokenPair)
def login(body: LoginBody, user_agent: str = Header(default="unknown")) -> TokenPair:
    # User-Agent 只作为设备会话元数据，不参与验证码验证。
    return _map_auth_error(lambda: login_with_code(body.phone, body.code, client_id=body.client_id, user_agent=user_agent))


@router.post("/refresh", response_model=TokenPair)
def refresh(body: RefreshBody) -> TokenPair:
    # Refresh Token 明文仅透传服务层并立即摘要匹配。
    return _map_auth_error(lambda: refresh_access_token(body.refresh_token))


@router.post("/logout")
def logout(body: LogoutBody) -> dict[str, bool]:
    # 登出成功后返回固定结构，不暴露会话状态细节。
    _map_auth_error(lambda: revoke_refresh_token(body.refresh_token))
    return {"revoked": True}
