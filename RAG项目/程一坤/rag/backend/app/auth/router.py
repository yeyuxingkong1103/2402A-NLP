"""认证 HTTP 接口。"""

import logging

from fastapi import APIRouter, Request

from app.auth.current_user import CurrentUser
from app.auth.schemas import (
    CodeRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
)
from app.auth.mailer import create_mailer
from app.auth.service import AuthService
from app.core.config import settings
from app.errors import bad_request_error, unauthorized_error


logger = logging.getLogger("app.auth")

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


_auth_service: AuthService | None = None


def _get_auth_service() -> AuthService:
    """获取认证服务单例（用户主体落 MySQL，会话令牌走 Redis）。"""
    global _auth_service
    if _auth_service is None:
        from app.auth.sql_store import SqlAuthStore
        from app.chat.chat_runtime import build_default_session_factory
        from app.db.redis_client import get_redis_client
        from app.auth.session_store import SessionStore

        redis_client = get_redis_client()
        session_store = SessionStore(redis_client, settings.session_ttl_seconds)
        # 用户主体落库（批次5）：注册/登录/改密走 MySQL users 表，重启不丢用户；
        # 会话工厂复用 chat_runtime 的装配，连接参数只维护一份
        auth_store = SqlAuthStore(build_default_session_factory())
        # mailer 统一使用 SMTP：注册验证码必须发送到用户邮箱，不在接口中回显
        _auth_service = AuthService(
            store=auth_store,
            session_store=session_store,
            mailer=create_mailer(settings),
        )
    return _auth_service


@router.post("/register/code", response_model=TokenResponse)
def send_register_code(request_body: CodeRequest, request: Request) -> TokenResponse:
    """发送注册验证码到用户邮箱。"""
    service = _get_auth_service()
    service.send_code(str(request_body.email), "register")
    logger.info("注册验证码发送请求完成: email=%s", request_body.email)
    data: dict = {"status": "sent"}
    return TokenResponse(
        code=0,
        message="验证码已发送",
        data=data,
        request_id=request.headers.get("X-Request-ID", ""),
    )


@router.post("/password-reset/code", response_model=TokenResponse)
def send_reset_code(request_body: CodeRequest, request: Request) -> TokenResponse:
    """发送密码重置验证码到用户邮箱。"""
    service = _get_auth_service()
    code = service.send_code(str(request_body.email), "reset")
    logger.info("密码重置验证码发送请求完成: email=%s", request_body.email)
    # development 回显验证码便于本地联调（production 用 SMTPMailer 真发信，不回显）
    data: dict = {"status": "sent"}
    if settings.environment == "development":
        data["debug_code"] = code
    return TokenResponse(
        code=0,
        message="验证码已发送",
        data=data,
        request_id=request.headers.get("X-Request-ID", ""),
    )


@router.post("/register", response_model=TokenResponse)
def register(request_body: RegisterRequest, request: Request) -> TokenResponse:
    """用户注册并返回访问令牌。"""
    service = _get_auth_service()
    try:
        token = service.register(
            str(request_body.email), request_body.password, request_body.code
        )
    except ValueError as error:
        raise bad_request_error(str(error)) from error
    logger.info("用户注册成功: email=%s", request_body.email)
    return TokenResponse(
        code=0,
        message="注册成功",
        data={"access_token": token, "token_type": "bearer"},
        request_id=request.headers.get("X-Request-ID", ""),
    )


@router.post("/login", response_model=TokenResponse)
def login(request_body: LoginRequest, request: Request) -> TokenResponse:
    """用户登录并返回访问令牌。"""
    service = _get_auth_service()
    try:
        token = service.login(str(request_body.email), request_body.password)
    except PermissionError as error:
        raise unauthorized_error(str(error)) from error
    logger.info("用户登录成功: email=%s", request_body.email)
    return TokenResponse(
        code=0,
        message="登录成功",
        data={"access_token": token, "token_type": "bearer"},
        request_id=request.headers.get("X-Request-ID", ""),
    )


@router.post("/password-reset", response_model=TokenResponse)
def reset_password(
    request_body: ResetPasswordRequest, request: Request
) -> TokenResponse:
    """验证验证码并重置用户密码。"""
    service = _get_auth_service()
    try:
        service.reset_password(
            str(request_body.email), request_body.password, request_body.code
        )
    except ValueError as error:
        raise bad_request_error(str(error)) from error
    logger.info("密码重置成功: email=%s", request_body.email)
    return TokenResponse(
        code=0,
        message="密码重置成功",
        data={"status": "reset"},
        request_id=request.headers.get("X-Request-ID", ""),
    )


@router.post("/logout", response_model=TokenResponse)
def logout(current_user: CurrentUser, request: Request) -> TokenResponse:
    """用户注销当前会话。"""
    service = _get_auth_service()
    # 从依赖注入获取的 session_user 中提取 user_id，重新读取令牌
    # 注意：CurrentUser 依赖已校验令牌有效性，此处直接注销
    authorization = request.headers.get("authorization", "")
    token = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    if token:
        service.logout(token)
        logger.info("用户注销成功: user_id=%s", current_user.user_id)

    return TokenResponse(
        code=0,
        message="注销成功",
        data={"status": "logged_out"},
        request_id=request.headers.get("X-Request-ID", ""),
    )
