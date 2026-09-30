"""FastAPI 依赖：从请求头取令牌并校验会话。"""

from typing import Annotated

from fastapi import Depends, Header

from app.auth.session_store import SessionStore, SessionUser
from app.db.redis_client import get_redis_client
from app.core.config import settings
from app.errors import unauthorized_error


def get_current_user(authorization: str | None = Header(default=None)) -> SessionUser:
    """从 Authorization 头解析 Bearer 令牌并校验会话。

    失败时统一抛出 unauthorized_error，返回错误码 40100。

    参数：
    - authorization: Authorization 请求头，格式为 "Bearer <token>"

    返回：
    - SessionUser: 会话用户信息

    异常：
    - HTTPException: 未登录或令牌无效时抛出 401
    """
    # 检查请求头是否存在
    if not authorization:
        raise unauthorized_error("未提供认证令牌")

    # 检查格式是否为 "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise unauthorized_error("认证令牌格式错误")

    token = parts[1]

    # 从 Redis 读取会话
    try:
        redis_client = get_redis_client()
        session_store = SessionStore(redis_client, settings.session_ttl_seconds)
        session_user = session_store.read_session(token)
    except Exception:
        # Redis 异常统一返回未授权，不得放行
        raise unauthorized_error("会话校验失败")

    # 会话不存在或已过期
    if session_user is None:
        raise unauthorized_error("会话已过期或无效")

    return session_user


# 依赖注入类型别名
CurrentUser = Annotated[SessionUser, Depends(get_current_user)]
