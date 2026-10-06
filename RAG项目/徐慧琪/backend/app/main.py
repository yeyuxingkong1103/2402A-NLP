"""应用装配：lifespan 装配重资源、中间件、异常映射与健康检查。

存在的理由（设计 §三）：本模块是**唯一**知道「服务由哪些部件组成」的地方 ——
重资源只在 lifespan 装一次存进 app.state（定调第 2 条），路由从 app.state 取。
它只做 HTTP 关注点（参数校验、鉴权、序列化、异常→状态码），不写业务：业务在
Answerer 里已经编排好了（定调第 1 条，也是本设计不加 service 层的原因）。

任务 3 交付四件：中间件（请求 ID、体大小上限）、异常映射（§七 那七行）、
lifespan 装配与释放、/healthz。业务路由（4~7）、鉴权依赖（4）、审计写入（7）
**不在本模块**，给它们留的接缝是：create_app 里注册中间件与处理器的那几行、
errors.py 的类型表、request_id 写进 scope 的那个键 —— 限流（任务 6）也按这条接，
本模块只挂一行中间件。

**装配的是公众侧超集**：本服务两侧都答（设计 §二 第 1 条），而 Answerer 的侧别
是**逐请求参数**（`answer(question, side)`），所以一套装配就能服务两侧 —— 区块
走不走由 Answerer._extras_for 按当次的 side 判（律师侧恒不走，③b-1 已钉住）。
反过来只装律师侧，公众侧就永远没有律师与费用区块（设计 §四 的契约要求有）。
代价**已知并接受**：with_extras 在启动时要 DeepSeek 密钥（公众侧问答本就需要它），
并且启动时会建一次 fee_log 留痕表。
"""
from __future__ import annotations

import contextlib
import functools
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

# 路由清单收在 api/ 包的 install_routes 里（本文件已到行数闸门边缘，且「有哪些
# 路径」该有单一落点）：这里只调用，不逐条 include_router
from app.api import install_routes
from app.core import audit, errors, factory, metrics, ratelimit, request_id, security
from app.core.security import AuthError
from app.db.milvus import COLLECTION as LAW_COLLECTION
from app.generation.profiles import SIDE_PUBLIC

logger = logging.getLogger(__name__)

# 请求体上限（设计 §六 防滥用表）。64KiB 是「粗筛」量级：问句与将来的会话上下文
# 远小于它，而整包上传、灌垃圾请求在这层就被挡掉。它不是问句长度上限 ——
# 那条（超限拒绝而不是截断）是任务 4 的 schema 职责，两层各挡各的
MAX_BODY_BYTES = 64 * 1024


class BodySizeLimitMiddleware:
    """卡 Content-Length 上限，超限当场 413（设计 §六 防滥用表）。

    选 413 而不是 400 的理由：RFC 9110 给「服务端不愿处理这么大的请求体」的就是
    413，客户端据此能分开「我发多了」与「我发的 JSON 是坏的」（后者才是 §七 那行
    的 400），运维看状态码也能一眼分流。代价是 §七 表里没有这一行，本任务把它
    补进 errors.PUBLIC_ERRORS 并在交付说明里写明。

    **已知缺口**：分块传输（无 Content-Length）不走这条判定。补它要读 body 计数，
    而读了就得把流重新拼给下游（多一次拷贝与一类新的出错方式）。后手有两道：
    任务 4 的问句长度上限按**字符数**拒绝（分块发送也拦得住），上线前由 Nginx 的
    client_max_body_size 兜底（设计 §六 已把 Nginx 层列为部署阶段的事）。
    """

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if _declared_length(scope) > self.max_bytes:
            code, message = errors.public_error(413)
            response = JSONResponse(status_code=413, content={
                "request_id": request_id.scope_request_id(scope),
                "error": {"code": code, "message": message}})
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _declared_length(scope: dict) -> int:
    """读 Content-Length；头缺失或不是数字都当 0（= 不限制）。

    非数字不当错误处理：畸形头在 ASGI 服务器（h11）那层就被拒了，能走到这里的
    只剩合法值；真遇到怪值时放行比拒绝稳妥 —— 这一层是防滥用，不是校验层。
    """
    raw = dict(scope.get("headers") or []).get(b"content-length", b"")
    try:
        return int(raw)
    except ValueError:
        return 0


def _error_response(request: Request, status_code: int, *, public_message: str = "",
                    headers: dict | None = None) -> JSONResponse:
    """按 §七 的统一形状出错误体：{"request_id", "error": {code, message}}。

    文案默认**只**从 errors.PUBLIC_ERRORS 按状态码查表；public_message 只给
    「设计明确要求带原文」的那一行（501，见 errors.FeatureNotImplemented）用。
    这个口子收在一个关键字参数上而不是「谁都可以传 message」：想泄露原文，
    得先改这个函数或显式打开异常类型上的 public_detail 开关。
    """
    code, message = errors.public_error(status_code)
    # 错误响应**自己带头**，不能只靠请求 ID 中间件：500 那条路径由
    # ServerErrorMiddleware 下发，而它在所有用户中间件**外面** —— 中间件没有机会
    # 给 500 挂头（这是实测出来的，不是推理），而 500 恰恰是最需要按 id 捞日志的
    # 一类。中间件那边因此改成「没有才补」，不会出现两个同名头
    rid = request_id.scope_request_id(request.scope)
    return JSONResponse(status_code=status_code,
                        headers={**(headers or {}), request_id.HEADER: rid},
                        content={"request_id": rid,
                                 "error": {"code": code,
                                           "message": public_message or message}})


def _on_api_error(request: Request, exc: errors.ApiError) -> JSONResponse:
    """自家异常：状态码由类型带（§七 表在 errors.py），文案按表查。"""
    headers = None
    if isinstance(exc, errors.RateLimited):
        # §六：限流与并发超限要带 Retry-After，客户端据此退避而不是立刻重试
        headers = {"Retry-After": str(exc.retry_after)}
    # 日志等级按「是不是我们的锅」分：4xx 是调用方的事（warning），5xx 要能
    # 在日志里一眼看见（error）。exception 原文只出现在这里，不出现在响应里
    logger.log(logging.WARNING if exc.status_code < 500 else logging.ERROR,
               "接口错误 %s %s → %d：%s", request.method, request.url.path,
               exc.status_code, exc.detail)
    # 只有显式打开开关的类型（501）才把 detail 交出去；别的类型交空串，
    # 让 _error_response 退回去查表 —— 判断收在这一行，不是散在各异常类型里
    message = exc.detail if exc.public_detail else ""
    return _error_response(request, exc.status_code, public_message=message,
                           headers=headers)


def _on_auth_error(request: Request, exc: AuthError) -> JSONResponse:
    """未认证 / 越权一律 404（设计 §二 第 9 条：不暴露路径存在性）。

    任务 2 把三类 token 错误做成 AuthError 的兄弟，正是为了这里只写一次映射；
    若改成 401/403，等于对未认证的探测者承认「这个路径存在」。
    """
    logger.warning("未认证/越权 %s %s：%r", request.method, request.url.path, exc)
    return _error_response(request, 404)


def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """参数校验失败 → 400（FastAPI 默认是 422，§七 第 1 行要的是 400）。

    只记字段位置与错误类型，不记输入值：公众侧的输入值就是提问原文，而设计 §五
    连审计都不记公众侧提问，日志这条旁路更不该记；何况畸形输入往往是超长串，
    整段进日志等于给灌日志的人递工具。
    """
    logger.warning("参数校验失败 %s %s：%s", request.method, request.url.path,
                   [(e.get("loc"), e.get("type")) for e in exc.errors()])
    return _error_response(request, 400)


def _on_http_error(request: Request, exc: HTTPException) -> JSONResponse:
    """框架抛的 HTTPException（未知路径 404、方法不允许 405）也走统一形状。

    同样不把 exc.detail 放进响应：框架与将来的路由都可能往里塞具体信息
    （如「用户 X 不存在」），统一文案把这类泄露收成一处。

    状态码先过 errors.public_status（405 → 404，连同 Allow 头的丢弃，理由与代价都在那边）。
    """
    status = errors.public_status(exc.status_code)
    logger.info("框架 HTTP 异常 %s %s → %d（原 %d）：%s", request.method,
                request.url.path, status, exc.status_code, exc.detail)
    return _error_response(request, status, headers=None if status != exc.status_code
                           else getattr(exc, "headers", None))


def _on_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """兜底 500：通用文案，堆栈只进日志（§七 第 7 行）。

    **exc_info=exc 而不是 logger.exception**：实测发现本处理器拿不到异常上下文
    （框架调用它时已不在 except 块里），logger.exception 打出来的是
    "(NoneType: None)" —— 堆栈静默丢掉，日志里只剩一句话，而这正是兜底处理器
    唯一的价值。传实例则无论上下文在不在都能取到堆栈（logging 会用它自带的
    __traceback__ 拼）。

    另注意：本处理器由 ServerErrorMiddleware 调用，出完响应后**仍会把异常重抛**
    （Starlette 的行为），所以 TestClient 默认参数下用例看到的是异常而不是 500
    响应 —— 要断言响应体得传 raise_server_exceptions=False（见测试注释）。

    **request_id 显式带上**（复审修的 Important）：记录工厂是从 contextvar 取 ID 的，
    而这里由 ServerErrorMiddleware 调用 —— 它在所有用户中间件**外面**，
    RequestIdMiddleware 的 finally: reset 早已跑过。实测同一请求头/体都带 id、这条
    日志却是 "-"，而 500 是唯一带堆栈的一类，用户报一个 id 却 grep 不到。故 ID 一律
    从 scope 取（不认 contextvar 里的残留值），再放回 contextvar 让记录工厂带上它，
    打完立刻复位。不能用 extra={"request_id": rid}：工厂已设过该属性，logging 会抛
    KeyError（不是覆盖，是当场报错）。

    **500 的指标也在这里记**（任务 8）：本处理器由 ServerErrorMiddleware 调用，它在
    所有用户中间件外面 —— 计数中间件根本没有机会看见这条响应。计入点收在这一行，
    而不是「指望最外层中间件兜住」（那是数不到的假接线，core/metrics 的注释同此）。
    """
    metrics.record_unhandled(request)
    token = request_id.set_current(request_id.scope_request_id(request.scope))
    try:
        logger.error("未处理的异常 %s %s：%r", request.method, request.url.path, exc,
                     exc_info=exc)
    finally:
        request_id.reset(token)
    return _error_response(request, 500)


def _install_handlers(app: FastAPI) -> None:
    """把 errors.PUBLIC_ERRORS 那张表挂到框架上（注册顺序无关，**覆盖的种类**才要紧）。

    五种注册对应 §七 的七行：ApiError（400/429/501/503）、AuthError（404）、
    RequestValidationError（400）、HTTPException（框架的 404/405 等）、
    Exception（500 兜底）；200 那行不是异常，落点是 errors.raise_if_answer_failed。
    """
    app.add_exception_handler(errors.ApiError, _on_api_error)
    app.add_exception_handler(AuthError, _on_auth_error)
    app.add_exception_handler(RequestValidationError, _on_validation_error)
    app.add_exception_handler(HTTPException, _on_http_error)
    app.add_exception_handler(Exception, _on_unexpected)


def _probe_mysql(conn) -> str:
    """MySQL 只读探针：确认连得上**且**业务表在。

    不用 `SELECT 1`：它只证明连接活着，指向空库/错库时照样绿 —— 而那正是 healthz
    该挡的情形。不用 COUNT(*)：健康检查会被反复打，1260 行的全表计数不该出现在
    这条路径上，LIMIT 1 足够回答「表在不在」。
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM article LIMIT 1")
        cur.fetchone()
    return "ok"


def _probe_milvus(client) -> str:
    """Milvus 只读探针：确认连得上**且**法条集合在（同上，不做 count 查询）。"""
    return "ok" if client.has_collection(LAW_COLLECTION) else "down"


def _run_probe(name: str, probe) -> str:
    """跑一个只读探针，收敛成 "ok"/"down"；异常只进日志，绝不进响应体。

    探针失败**不是**请求失败：healthz 的职责是把结果如实报出来（503 + 哪个依赖
    down），而不是把它变成一个 500 —— 那会让「依赖挂了」与「healthz 自己坏了」
    在监控上不可区分。
    """
    try:
        return probe()
    except Exception as exc:  # noqa: BLE001 —— 见 docstring：探针异常不改变接口形态
        logger.error("healthz 探针失败：%s：%r", name, exc)
        return "down"


def healthz(request: Request) -> JSONResponse:
    """连通性检查（设计 §四）。

    查的是 app.state 里那套**真在用的**连接，不是新开的：新开一个只能证明「库还
    在」，证明不了「服务手里那套连接还能用」—— 而后者才是「这个实例该不该被摘掉」
    的判据（MySQL 的 wait_timeout 到期后正是这种形态：库活着、实例的读路径已断）。

    响应体是 healthz 自己的契约（要能一眼看出哪个依赖挂了），不是 §七 的
    错误形状；它不含任何异常原文（探针的异常只在日志里），故与「通用文案」那条
    的立意一致。依赖不可用 → 503，让编排层据此摘流量。
    """
    services = getattr(request.app.state, "services", None)
    if services is None:
        # 装配未完成或启动失败：两个依赖都算不可用。失败方向朝下 —— 这里若报 ok，
        # 监控会把一个空壳实例当成健康的
        checks = {"mysql": "down", "milvus": "down"}
    else:
        checks = {
            "mysql": _run_probe("mysql",
                                functools.partial(_probe_mysql, services.conn)),
            "milvus": _run_probe("milvus",
                                 functools.partial(_probe_milvus, services.client)),
        }
    healthy = all(value == "ok" for value in checks.values())
    return JSONResponse(status_code=200 if healthy else 503, content={
        "request_id": request_id.scope_request_id(request.scope),
        "status": "ok" if healthy else "down", "checks": checks})


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """启动装一次重资源、关闭释放一次（设计 §三 定调第 2 条）。

    装配是同步阻塞的（加载模型要几十秒），这里直接调而不是丢线程池：启动期还没有
    请求在跑，阻塞事件循环没有代价，多一层抽象只会多一种出错方式。装配失败时
    **不**在 app.state 留半个 Services（build_services 自己回收了半成品，见
    core/factory.release），异常继续往上抛 —— 启动失败必须是响亮的，不能带着空壳
    起服务（那会让每个请求都在业务代码里炸出 500，而根因在启动日志里）。
    """
    # 缺签名密钥要让服务起不来（任务 2 交接、终审 I-5），且放在装配**之前**：
    # 缺配置时不该先花几十秒加载模型。MissingSecretError 原样上抛，不伪装成
    # 401/500（那会把「这台机器配置错了」说成「你没登录」）；密钥仍只由
    # security.load_secret 读，这里只是把读取提前到启动点
    security.load_secret()
    services = app.state.services_factory()
    app.state.services = services
    logger.info("重资源装配完成，服务就绪")
    try:
        yield
    finally:
        # 先清 app.state 再 close：这个瞬间进来的 healthz 会看到「没有 services」
        # （= down），而不是一套正在关闭的连接 —— 后者会报出随机结果
        app.state.services = None
        services.close()
        logger.info("重资源已释放")


def create_app(*, services_factory=None) -> FastAPI:
    """装配应用。参数只为测试注入装配器，生产走默认值；自省文档三件套全关（理由见下）。"""
    request_id.install()
    # 自省文档全关：schema 会列出全部已注册路径（含律师侧三条），抵消 §二 第 9 条的一半
    app = FastAPI(title="法律条文检索与公众咨询", lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)
    # 装配器收成**零参**可调用体：lifespan 因此不必知道 side/with_extras 这些
    # 装配参数，测试注入的替身也不必假装接受它们
    app.state.services_factory = services_factory or functools.partial(
        factory.build_services, SIDE_PUBLIC, with_extras=True)
    # 指标注册表每 app 一个（不是模块级单例：单例会让测试与多 app 场景互相污染，
    # core/metrics.Registry 的注释同此）；中间件与 /metrics 端点从同一处取
    app.state.metrics = metrics.Registry()
    # 体上限加在前（内层）、请求 ID 加在后（外层）：后加的在外。413 与 500 也要带
    # request_id，把 ID 放内层正好会让最需要 id 的两类响应缺了它
    app.add_middleware(BodySizeLimitMiddleware)
    # 限流/审计夹在两者之间（任务 6/7）：都在请求 ID 之内（429 与审计写失败的告警
    # 都要带 id）；限流在体上限之外（要计体超限的尝试），审计在限流之外（429 入账）
    app.add_middleware(ratelimit.RateLimitMiddleware)
    app.add_middleware(audit.AuditMiddleware)
    # 计数（任务 8）在请求 ID 之内、审计与限流**之外**：413/429 这类由更内层中间件
    # 自己出的响应必须经过它才算得进来（在它们之内就数不到）；500 不经任何用户中间件
    # —— 那一条的计入点在 _on_unexpected（见那里与 core/metrics 的注释）
    app.add_middleware(metrics.MetricsMiddleware, registry=app.state.metrics)
    app.add_middleware(request_id.RequestIdMiddleware)
    _install_handlers(app)
    # healthz 走 add_api_route 而不是 @app.get 装饰器：路径的注册点全部收在这个
    # 函数里（任务 4~7 的 include_router 也排队在此），实测时能一眼看全有哪些路径
    app.add_api_route("/healthz", healthz, methods=["GET"])
    # /metrics 与 healthz 并列（内网端点，设计 §四）：处理函数在 core/metrics 里，
    # 本文件只注册；不计入自身指标的口径见那边的 EXCLUDED_PATHS
    app.add_api_route("/metrics", metrics.endpoint, methods=["GET"])
    # 业务路由在 healthz 之后注册：顺序只影响 OpenAPI 文档里的排列，不影响匹配
    install_routes(app)
    return app


# uvicorn 的入口：`uvicorn app.main:app --app-dir backend`。这里只建对象，重资源
# 要等 lifespan 才装 —— 导入本模块（含测试收集）不会加载模型
app = create_app()
