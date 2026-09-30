"""FastAPI 的依赖包装：请求 ID、当前用户、角色门、共享连接。

存在的理由（设计 §三 的模块表把「RBAC 依赖」列在 security.py，实际落在 api/）：
`core/security.py` 是**纯函数与异常**（它的 docstring 写明「Depends 包装在 api/ 侧」），
因为它不该知道 HTTP、也不该把 2GB 模型加载拖起来。本模块因此只做三件事：
从请求里取 token、调 security 的纯函数、把失败翻译成 core/errors 的类型 ——
不复制任何鉴权判定（判定只有一处，就是 security.py）。

**三条失败语义**（设计 §七 + §二 第 9 条）：
  - 缺 token / token 过期 / 签名不符 / 账号停用 → `AuthError` 家族 → **404**。
    404 而不是 401：未认证的访问者不该知道这个路径存在（§二 第 9 条）。
    公开的 /auth/login 不在这一档 —— 它的失败是 401（见 errors.Unauthorized）。
  - 没配签名密钥 → `MissingSecretError` **原样上抛** → 500。它是服务端配置
    故障，伪装成 404 会把运维引到「鉴权」这条完全错误的路上（任务 2 的裁决）。
  - 装配未完成（app.state 里没有 services）→ 503。那是启动失败，不是「你没登录」。
"""
from __future__ import annotations

from fastapi import Depends, Request

from app.core import errors, request_id, security
from app.core.security import CurrentUser, Role
from app.db import users

# 认证方案前缀。RFC 6750 说方案名大小写不敏感，故比较时统一小写；
# 但**只认这一个方案** —— 收下 Basic/Digest 意味着一堆我们没实现的语义
BEARER_SCHEME = "bearer"


def request_id_of(request: Request) -> str:
    """本次请求的 ID（响应体与日志对齐用）。键名只在这里出现一处。

    取 scope["state"] 而不是 contextvar：错误响应（如 404）由异常处理器同一条
    请求链出，两者共享同一个 scope；而 contextvar 在 500 那条路径上已被复位
    （任务 3 踩过：日志记成 "-"）。缺值时给 NO_REQUEST —— 缺 id 的响应仍是
    合法响应，缺 id 的日志仍可读（与 main._scope_request_id 同口径）。
    """
    scope_request_id = (request.scope.get("state") or {}).get(request_id.STATE_KEY)
    return scope_request_id or request_id.NO_REQUEST


def bearer_token(request: Request) -> str:
    """从 Authorization 头取 token；缺失或形状不对当场抛 AuthError（→404）。

    形状不对也走同一个异常：告诉调用方「你少写了 Bearer」与「你 token 是坏的」
    对探测者是两种信息，而这一层的立场是**一种都不给**。
    """
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != BEARER_SCHEME or not token:
        raise security.TokenInvalidError("缺少或畸形的 Authorization 头")
    return token


def current_user(request: Request) -> CurrentUser:
    """鉴权依赖：Bearer token → 当前用户上下文（每个受保护路由都挂它）。

    **逐请求查一次 is_active**（任务 2 复审的交接、本任务的落点）：token 校验
    只看签名与 exp，若不查库，「停用」的生效窗口就是 token 剩余有效期（最长
    8 小时）。停用的真实触发点是离职与设备丢失，恰恰最需要立刻生效。

    连接策略（本任务的结论，**实测推翻了两条候选**）：用 `Services.account_conn`
    —— 一条**专用的、autocommit 的、长生命周期**的连接。三条路各自为什么不成：
      - **复用业务连接**（最初的方案）：实测不成立，不是「慢一点」而是**读不到**。
        pymysql 默认 autocommit=False，业务连接的第一次读（启动时的 healthz 探针）
        就开启事务并定下 REPEATABLE READ 读视图，此后**运维在别处提交的停用它
        永远看不见**（实测：停用后仍读到 True，直到该连接 commit 一次）。后果是
        这条检查形同虚设，且形态是「静默失效」——而它要挡的恰恰是离职与设备丢失。
      - **逐请求 connect()**：多付一条 TCP 会话 + caching_sha2 认证握手，并发 20
        （设计 §六）时有 20 条额外连接且**生命周期无人负责**（谁关？请求异常时？）
        —— 那正是任务 3 修掉的「连接被留在损坏状态」同族问题。
      - **短 TTL 缓存**（哪怕 5 秒）：把这条检查要消灭的那个窗口重新打开，还要
        另一套失效逻辑（运维停用后缓存还在 = 停用不即时）。
    专用连接的成本是**进程内多一条连接**（不是每请求一条），换来的是：读视图
    逐语句新建（登录刚建好的账号与刚下的停用都立刻可见）、与业务查询**不争锁**
    （队头阻塞随之消失）、且写入语义的变化被隔离在这条只读的连接上（见
    db/mysql.connect_autocommit 末尾的告诫）。
    """
    user = security.verify_token(bearer_token(request), secret=security.load_secret())
    # 查库异常**不在这里吞**：连不上库是服务故障，翻成 404 会把「我们坏了」
    # 说成「你没登录」（与 MissingSecretError 不伪装成未登录同一条口径）
    if not users.is_active(account_conn(request), user.id):
        # 账号不存在与账号停用对外的差别本身就是信息（「这个人存在过」），
        # 故两者同一种响应；PermissionDeniedError 属 AuthError 家族 → 404
        raise security.PermissionDeniedError(f"账号不可用：id={user.id}")
    return user


def require_roles(*roles: Role):
    """角色门：造一个「当前用户必须属于 roles」的依赖（设计 §三 的 RBAC）。

    用法：`user: CurrentUser = Depends(require_roles(Role.PARTNER))`。
    角色不够抛 PermissionDeniedError（AuthError 的一种）→ 404：与未认证同一种
    响应。若回 403，探测者就能把「路径存在且我角色不够」从 404 里分出来 ——
    那是比「路径存在」更细的一份情报。
    """
    allowed = frozenset(roles)

    # 形参默认值必须是 Depends(...) 实例：FastAPI 只认它，写成一个裸函数
    # 会被当成**默认值**（闭包原样收到函数对象），鉴权静默失效且不报错
    def gate(user: CurrentUser = Depends(current_user)) -> CurrentUser:
        security.require_role(user, allowed)
        return user

    return gate


def services_of(request: Request):
    """取 lifespan 装配好的一套重资源（设计 §三 定调第 2 条：绝不在路由里装配）。

    每个用到重资源的端点都走这里，于是「装配未完成时怎么办」只有一处答案：
    抛 503。写 `getattr(request.app.state, "services", None)` 的各路由各判一次的话，
    总有一条会写成 `services.answerer` 直接 AttributeError → 500 —— 那是把
    「服务没起来」说成「服务内部错误」，运维看状态码就得不出结论。
    """
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise errors.ServiceUnavailable("重资源未装配：端点无法工作")
    return services


def account_conn(request: Request):
    """取账号查询用的连接（策略与实测见 current_user）：装配好的专用 autocommit 连接。

    登录端点与鉴权依赖**共用这一个出口**：两者要的是同一个性质（读到最新提交），
    分成两处各取一次连接，迟早有一处退回业务连接 —— 而退回后的表现是登录查不到
    刚建的账号，看起来与「口令错」一模一样。

    没有 services 说明启动失败或尚未启动：这条路径上判 503 而不是 404 ——
    「我们坏了」与「你没登录」在运维与用户两侧都是完全不同的结论（§七 的
    503/200 同族口径）。此刻 token 已经校验过，所以能走到这里的一定是
    「身份合法但查不动库」，503 是诚实的那一个。
    """
    conn = services_of(request).account_conn
    if conn is None:
        # Services 少填这一格时的兜底：AttributeError 也会变成 500，但那条
        # 报错长得像代码 bug，而真实语义是「这条资源没装配」——503 才对得上
        raise errors.ServiceUnavailable("账号连接未装配：无法核对账号")
    return conn
