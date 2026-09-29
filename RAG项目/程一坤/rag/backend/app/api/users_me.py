"""当前登录用户信息接口（批次 16-A，接口文档 3.8）。

语义（按契约）：
- 只返回当前登录用户自己的信息，不接受任何 user_id 参数；
- 未登录 → 401（40100，认证依赖拦截）；
- 普通用户也要能正常拿到 is_admin=false，不因"没有 admin 数据"报错；
- 不得返回密码哈希、会话令牌等敏感字段。

用途：前端据此判断是否显示"审核"入口、以及记忆开关的初始状态。
"""

from fastapi import APIRouter, Request

from app.auth.current_user import CurrentUser

router = APIRouter(tags=["users"])


def _get_session_factory(request: Request):
    """users 表会话工厂（测试可经 app.state.users_me_session_factory 注入 SQLite）。"""
    factory = getattr(request.app.state, "users_me_session_factory", None)
    if factory is None:
        from app.chat.chat_runtime import build_default_session_factory

        factory = build_default_session_factory()
    return factory


@router.get("/api/v1/users/me")
def get_current_user_info(current_user: CurrentUser, request: Request) -> dict:
    """返回当前登录用户信息（契约 3.8）。

    is_admin 取认证会话（登录时落定，会话期内权威）；
    email / long_term_memory_enabled 取 users 表当前值；
    会话有效但用户行已被删除的边缘场景：返回空邮箱与关闭态，不报 500。
    """
    from sqlalchemy import select

    from app.db.sql_models import User

    with _get_session_factory(request)() as session:
        user = session.scalar(select(User).where(User.user_key == current_user.user_id))

    return {
        "code": 0,
        "message": "success",
        "data": {
            "user_id": current_user.user_id,
            "email": user.email if user else "",
            "is_admin": bool(current_user.is_admin),
            "long_term_memory_enabled": bool(user.long_term_memory_enabled)
            if user
            else False,
        },
        "request_id": request.headers.get("X-Request-ID", ""),
    }
