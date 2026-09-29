"""API 依赖：JWT 鉴权、管理员校验、客户端信息、Token 吊销。

依赖注入（Depends）是 FastAPI 的核心机制：路由函数把"如何得到当前用户"
这类通用逻辑声明为依赖，FastAPI 会在调用路由前自动执行它们，并把结果注入参数。
这样路由函数本身就可以保持极简，只关心业务转发。
"""
from typing import Optional

import jwt as pyjwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.core.exceptions import AuthError, PermissionError_
from src.core.logging import get_logger
from src.core.security import decode_token, revoke_token
from src.db import redis as redis_db
from src.db.mysql import get_db
from src.models import User
from src.services import persona_service, user_service

logger = get_logger("api.deps")

# HTTPBearer 负责从请求头 Authorization: Bearer <token> 中解析出凭证。
# auto_error=False：即使没有携带凭证也不立刻报错，而是交给下面的函数手动处理，
# 这样我们可以返回更友好的中文错误提示，并兼容"可选登录"（游客）的场景。
bearer_scheme = HTTPBearer(auto_error=False, description="JWT Bearer Token")


def get_client_ip(request: Request) -> str:
    """解析客户端真实 IP，供审计日志与限流使用。

    优先取 x-forwarded-for 头（代理/网关场景下客户端真实 IP 通常在这里），
    没有代理时再回退到直连的 client.host。
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """从 JWT 中解析并加载当前登录用户。

    这是所有"需要登录"接口的统一鉴权入口，承担三道关卡：
    1. 凭证存在性校验（未登录直接拒绝）；
    2. Token 合法性校验（过期、签名错误、类型不对、已被吊销）；
    3. 账号状态校验（用户是否存在、是否被禁用）。
    校验通过后把 User ORM 对象注入路由，路由因此能直接拿到"当前是谁"。
    """
    if credentials is None or not credentials.credentials:
        raise AuthError("缺少认证信息，请先登录")
    try:
        payload = decode_token(credentials.credentials)
    except pyjwt.ExpiredSignatureError:
        raise AuthError("登录已过期，请重新登录")
    except pyjwt.PyJWTError as exc:
        raise AuthError(f"Token 无效：{exc}")

    if payload.get("type") != "access":
        raise AuthError("Token 类型错误")

    # G10：已吊销（logout/改密）的 token 直接拒绝
    # 每次请求都查一次 Redis 黑名单，是"主动吊销"能即时生效的关键。
    if redis_db.is_token_blacklisted(payload.get("jti", "")):
        raise AuthError("登录已失效，请重新登录")

    user = db.get(User, int(payload.get("sub", 0)))
    if not user:
        raise AuthError("用户不存在")
    if user.status != 1:
        raise AuthError("账号已被禁用")
    return user


def revoke_current_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> bool:
    """吊销当前请求携带的 access token（logout / 修改密码后调用）。

    作为依赖注入路由，会在路由体执行前先完成吊销动作，
    并返回"是否吊销成功"的布尔值，供路由决定如何提示用户。
    """
    if credentials is None or not credentials.credentials:
        return False
    try:
        return revoke_token(decode_token(credentials.credentials))
    except pyjwt.PyJWTError:
        return False


def get_current_admin(user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)) -> User:
    """管理员鉴权依赖。

    它"依赖上面的依赖"（先 get_current_user 确认登录），再进一步判断管理员身份。
    这种依赖链是 FastAPI 常见写法：把权限校验叠加在身份校验之上，
    需要管理员权限的路由只要换成 Depends(get_current_admin) 即可。
    """
    if not persona_service.is_admin(db, user):
        raise PermissionError_("需要管理员权限")
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """可选登录：带了有效 Token 返回用户，否则返回 None（游客）。

    用于"登录/未登录都能访问"的接口，路由再根据返回值是否为 None 决定走哪条分支。
    这里特意吞掉 AuthError，是为了让无效凭证也不阻断游客访问。
    """
    if credentials is None:
        return None
    try:
        return get_current_user(credentials, db)
    except AuthError:
        return None