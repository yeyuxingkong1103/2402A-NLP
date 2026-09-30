"""src/schemas/auth.py —— 认证相关的请求/响应模型。

在链路中的位置：
    src/api/routers/auth.py 用它做入参校验（RegisterRequest / LoginRequest）和响应定型（TokenResponse）

这些模型的价值在于"把校验写进类型"：
    FastAPI 会自动拿它们校验请求体，不满足约束的请求在进入路由函数之前就被拦下并返回 422，
    路由代码里因此不需要写任何长度判断 —— 校验规则集中在一处，也不会被漏掉。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """注册请求体。

    字段：
        username: 3~128 字符。下限防无意义的一字符用户名，上限与 user 表的字段长度对齐
        password: 6~128 字符。下限是弱口令的最低门槛；
                  上限 128 是必要的 —— PBKDF2 对超长输入的计算代价与长度相关，
                  不限制长度的话，一个几 MB 的"口令"就能变成一次廉价的资源消耗攻击
    """

    username: str = Field(min_length=3, max_length=128)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    """登录请求体。

    字段：
        username / password: 这里**不做**长度校验，与注册的差异是有意的。

    为什么不校验：
        登录只负责"核对已有口令"，如果这里加了长度限制，
        早期用较短口令注册的账号就再也登不进来。
        校验属于注册环节的准入规则，不该出现在登录环节。
    """

    username: str
    password: str


class TokenResponse(BaseModel):
    """令牌响应体（注册和登录共用）。

    字段：
        access_token: JWT 字符串
        token_type:   固定 "bearer"，配合 HTTP 的 Authorization: Bearer <token> 用法
        user_id:      用户 id，方便前端直接拿到身份而不必解析 JWT
    """

    access_token: str
    token_type: str = "bearer"
    user_id: int
