"""`POST /api/v1/auth/login`：用户名口令 → JWT（设计 §四、§五）。

存在的理由：三层隔离与 RBAC 都建立在「你是谁」之上，而本轮不做管理界面
（设计 §一），故换取身份的唯一入口就是这个端点。它只做三件事：查一行账号、
验一次口令、签一个 token —— 判定全部来自 core/security.py，这里不复制任何
哈希或 token 逻辑（密码策略在运维脚本，口令机制在 security）。

**失败一律 401、且不区分原因**（本任务的红线）：口令不对、查无此人、账号停用
三种都从同一个出口抛 errors.Unauthorized，响应体逐字节相同。分开报等于告诉
一个没通过认证的人「这个用户名存在」——那正是撞库最想要的那一位信息。
安全点：没认证的人不该能从 /auth/login 得到任何比「失败」更多的信息。
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.api import deps, schemas
from app.core import errors, security
from app.db import users

router = APIRouter(prefix="/api/v1", tags=["auth"])

# 查无此人时拿来**陪跑一次 scrypt** 的假哈希（值本身没有意义，见下）。
# 它是本次生成的一个真实格式串：任何走 _parse 的合法串都要花掉与真实校验同样的
# 时间与内存 —— 这正是它存在的理由。
# 为什么需要它：直接 return 会让「用户不存在」比「口令错误」快约 40ms（一次
# scrypt 的量级），把账号存在性从**时间**上漏出去，而我们刚刚在上面的出口
# 花力气把这一点从**响应体**上抹掉。两者的挡法不同、缺一不可。
# 换掉这个字面量时必须仍是合法 scrypt 串（参数落在 security._SCRYPT_LIMITS 内），
# 否则 _parse 判 None → 立刻返回 → 陪跑静默失效：故测试里有一条钉住「它能被
# _parse 解析」，另有一条钉住「未知用户名这条路上 verify_password 确实被调用过」。
DUMMY_HASH = ("scrypt$16384$8$1$Eq5s8uTcf2jlnyE7vZArAQ=="
              "$dGUOKZyaB/XefFAinZ8ITbnl79hYRJdhQEjxvg1QQWg=")


@router.post("/auth/login", response_model=schemas.LoginResponse)
def login(body: schemas.LoginRequest, request: Request) -> schemas.LoginResponse:
    """校验口令并签发 token。

    token 有效期取 security.TOKEN_TTL_S（8 小时）：它同时是「停用账号的已签发
    token 的残留窗口」的**名义**值 —— 实际窗口已被鉴权依赖的逐请求查库压到即时
    （见 deps.current_user），所以这里的 TTL 只承担「一个工作日」的语义。
    """
    conn = deps.account_conn(request)
    row = users.get_by_username(conn, body.username)
    stored = row["password_hash"] if row is not None else DUMMY_HASH
    matched = security.verify_password(body.password, stored)
    # 先算再判：`row is not None and not matched` 写成短路求值也能过用例，
    # 但那样未知用户就少了一次 scrypt（时间侧信道重新打开）—— 判据与耗时
    # 必须解耦，故上面那两行**无条件**跑完
    if row is None or not matched or not row["is_active"]:
        # detail 只进日志（Unauthorized.public_detail 是关的）：它是排查
        # 「哪一种失败」的唯一线索，而对外文案由 errors.PUBLIC_ERRORS[401] 定。
        # 不含口令（任何形态）、不含哈希
        raise errors.Unauthorized(f"登录失败：username={body.username!r}")
    # 停用已在上面的条件下拦掉，故 from_row 不会在这里抛 PermissionDeniedError
    # （它内部的 is_active 判据与这里是同一列，不是第二份口径）
    user = security.CurrentUser.from_row(row)
    token = security.sign_token(user, secret=security.load_secret())
    return schemas.LoginResponse(access_token=token, role=user.role.value,
                                 team_id=user.team_id)
