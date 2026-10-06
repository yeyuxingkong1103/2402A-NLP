"""认证接口：注册、登录、刷新、登出、当前用户。

本模块只做"转发"：接收 Pydantic 请求模型，转交给 user_service 处理，
再用统一的 ok() 包装返回。真正的密码校验、JWT 签发、审计等逻辑都在服务层。
"""
from __future__ import annotations

import datetime as _dt

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from src.api.deps import get_client_ip, get_current_user, revoke_current_token
from src.core.config import settings
from src.core.exceptions import RateLimitError, ok
from src.db.mysql import get_db
from src.db.redis import check_rate_limit_by_key, login_rate_limit_key
from src.models import User
from src.schemas import (LoginRequest, RefreshRequest, RegisterRequest, UserInfo,
                         UserUpdateRequest)
from src.services import user_service

# prefix 表示该模块下所有路由都挂在 /auth 之下，tags 用于给 OpenAPI 文档分组。
router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", summary="用户注册")
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    # RegisterRequest 是 Pydantic 模型：FastAPI 会在进入函数前自动完成请求体反序列化与校验。
    # 这里只负责"搬运"字段给服务层，注册的具体规则（重名校验、密码加密等）在 user_service.register。
    user = user_service.register(
        db, payload.username, payload.password, payload.email, payload.phone, payload.nickname
    )
    # 注册属于敏感操作，写入审计日志，并记录来源 IP 便于追溯。
    user_service.add_audit_log(db, user.id, "register", f"username={user.username}", get_client_ip(request))
    return ok({"user_id": user.id, "username": user.username}, "注册成功")


@router.post("/login", summary="用户登录（JWT）")
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    # 阶段 4：未登录接口按 IP 限流，防暴力破解（登录失败本身已有审计与锁定语义）
    # 限流放在控制器这一层：因为它是"HTTP 接入层"的横切关注点，与业务无关，且要在进入业务前快速拦截。
    ip = get_client_ip(request)
    minute = _dt.datetime.now().strftime("%Y%m%d%H%M")
    if not check_rate_limit_by_key(login_rate_limit_key(ip, minute),
                                   settings.login_rate_limit_per_minute):
        raise RateLimitError("登录尝试过于频繁，请一分钟后再试")
    data = user_service.login(
        db, payload.username, payload.password,
        ip=ip, user_agent=request.headers.get("user-agent"),
    )
    return ok(data, "登录成功")


@router.post("/refresh", summary="刷新 Token")
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    # 刷新接口不需要登录态（因为 access token 可能已过期），
    # 因此它不依赖 get_current_user，而是直接校验 refresh_token 本身。
    return ok(user_service.refresh_token(db, payload.refresh_token), "刷新成功")


@router.post("/logout", summary="退出登录（吊销当前 Token）")
def logout(request: Request, user: User = Depends(get_current_user),
           revoked: bool = Depends(revoke_current_token),
           db: Session = Depends(get_db)):
    # 两个依赖配合：get_current_user 先确认"是谁"（未登录不能登出），
    # revoke_current_token 再吊销当前 token，并返回是否成功。
    user_service.add_audit_log(db, user.id, "logout", ip=get_client_ip(request))
    return ok({"token_revoked": revoked}, "已退出登录")


@router.get("/me", summary="当前用户信息", response_model=None)
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # response_model=None 表示响应结构由服务层的 ok() 动态决定，不强制走 Pydantic 序列化。
    return ok(user_service.get_profile(db, user.id))


@router.put("/me", summary="修改当前用户信息")
def update_me(payload: UserUpdateRequest, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    # exclude_none=True：只提交用户显式填写的字段，未填写的字段不覆盖原值（部分更新）。
    data = user_service.update_profile(db, user.id, payload.model_dump(exclude_none=True))
    return ok(data, "更新成功")