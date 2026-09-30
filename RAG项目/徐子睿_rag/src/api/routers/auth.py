"""src/api/routers/auth.py —— 注册与登录路由。

在链路中的位置：
    HTTP → 【本文件】 → src/api/deps.py（口令哈希 / JWT 签发）
                       → src/models/database.py（读写 user 表）

两个接口，路由前缀 /api/v1/auth：
    POST /register  注册并直接返回令牌（注册后免去再登录一次）
    POST /login     登录换令牌

令牌是后续所有业务接口的通行证：
    拿到 access_token 后，请求时带 Authorization: Bearer <token>，
    由 src/api/deps.py 的 current_user 解析出用户身份。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.api.deps import create_access_token, hash_password, verify_password
from src.models.database import User, db_session
from src.schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
def register(request: RegisterRequest):
    """注册新用户，成功即下发访问令牌。

    参数：
        request: RegisterRequest（username + password，长度等约束在 schema 里）
    返回：
        TokenResponse(access_token, user_id)。
    异常：
        用户名已存在 -> 409。

    先查重再插入，并用 409 Conflict 而不是 400：
        409 的语义正是"与现有资源冲突"，比 400（请求本身不合法）更准确，
        调用方能据此区分"这个名字被占了"和"你参数写错了"。

    这个"先查后插"存在极小的竞态窗口：
        两个请求同时通过查重、随后都插入，就会得到一个报错的 500
        （或在不设唯一约束时产生重复用户）。
        因此 User.username 上还建了 unique 索引作为最终防线 ——
        应用层查重是为了给出友好提示，数据库约束才是正确性的保证。

    db.refresh(user) 是必需的：
        user.id 由数据库自增生成，commit 前对象上是 None。
        不 refresh 就拿不到 id，create_access_token 会签出一个 sub=None 的坏令牌。

    注册成功直接返回令牌：
        少一次登录往返，也让"注册即用"的体验更顺。
    """
    with db_session() as db:
        exists = db.scalar(select(User).where(User.username == request.username))
        if exists:
            raise HTTPException(status_code=409, detail="用户名已存在")
        user = User(username=request.username, password_hash=hash_password(request.password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return TokenResponse(access_token=create_access_token(user), user_id=user.id)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    """登录校验并下发访问令牌。

    参数：
        request: LoginRequest（username + password）
    返回：
        TokenResponse(access_token, user_id)。
    异常：
        用户名不存在或口令错误 -> 401，且**两者返回同一句提示**。

    为什么"用户不存在"和"口令错误"要返回完全相同的消息：
        一旦两者措辞不同，攻击者就能用它来判断某个用户名是否已注册
        （用户枚举漏洞）。统一成"用户名或密码错误"不泄露任何信息。

    为什么用 `not user or not verify_password(...)` 合并判断：
        `or` 的短路特性让"用户不存在"时不再调用 verify_password
        （省一次无意义的口令哈希计算），同时两分支走同一条报错路径。
        这也让代码里不存在"因为用户不存在而提前返回"的差异分支 ——
        逻辑上就没有泄露信息的可能。
    """
    with db_session() as db:
        user = db.scalar(select(User).where(User.username == request.username))
        if not user or not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        return TokenResponse(access_token=create_access_token(user), user_id=user.id)
