"""FastAPI 应用入口。

启动流程：
1. 初始化日志与目录
2. 初始化 MySQL（建库建表）、幂等写入系统角色与三个心理医生角色
3. 初始化 Milvus Collection（知识库 + 长期记忆）
4. 预加载 BGE-M3 / BGE-Reranker-v2-M3（可通过 WARMUP_MODELS=0 关闭）

本文件只负责“装配”：把路由、中间件、异常处理、生命周期钩子拼成一个 App。
具体业务逻辑一律下沉到 src/api、src/services、src/db，保持入口文件薄而清晰。
"""
import time
# asynccontextmanager 把「进入/退出」两个时机包装成一个 async with 上下文：
# yield 之前的代码=启动时执行，yield 之后的代码=关闭时执行，
# 这正是 FastAPI lifespan 需要的形态（比老式的 @app.on_event 更安全，
# 因为它保证启动/关闭资源是成对出现、不会漏掉释放）。
from contextlib import asynccontextmanager
# uuid4 用于生成请求追踪 ID（见 access_log_middleware）。
from uuid import uuid4

from fastapi import FastAPI, Request
# RequestValidationError 是 Pydantic 参数校验失败时 FastAPI 抛出的异常类型，
# 单独捕获它才能把它翻译成项目统一的错误响应体，而不是 FastAPI 默认的 422 结构。
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.router import api_router
from src.core.config import settings
# AppError=业务可预期异常（带 code/http_status）；fail/ok 是统一的响应包装函数。
from src.core.exceptions import AppError, fail, ok
# set_request_id 把“当前请求 ID”写进上下文变量，使日志能跨函数贯穿同一条请求。
from src.core.logging import get_logger, set_request_id, setup_logging

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：yield 前=启动初始化，yield 后=停机清理。

    :param app: FastAPI 实例（此处未直接使用，由框架传入）
    """
    # 日志必须先初始化，否则后面所有 logger.info 都会丢格式/丢文件输出。
    setup_logging()
    logger.info("=" * 72)
    logger.info("启动 %s v%s", settings.app_name, settings.app_version)
    logger.info("=" * 72)

    # 0. 安全校验（S3/G8）：密钥缺失拒绝启动；调试模式监听公网给出警告
    # 这里的异常故意不捕获：安全配置错误属于“不可降级”的致命问题，
    # 必须让进程启动失败，避免带着弱密钥/空密钥对外提供服务。
    settings.validate_security()
    if settings.debug and settings.api_host == "0.0.0.0":
        logger.warning("DEBUG=true 且监听 0.0.0.0：仅建议本机开发使用，生产环境请关闭 DEBUG 并前置 HTTPS")

    # 1. MySQL
    # 顺序上先 MySQL 后 Milvus：MySQL 是元数据主库，角色的来源，
    # 且后续 Milvus 知识库需要 persona_id 才能有意义地对齐。
    try:
        # 延迟导入（放在 try 内）：即使某个依赖模块导入失败，也只是走到下面的
        # except 分支记日志，而不会让整个进程崩溃——这是“软降级”的第一层保护。
        from src.db.mysql import init_db, session_scope
        from src.services import persona_service, user_service
        # init_db() 内部会「建库 + 建表」，幂等操作，重复启动不会报错。
        init_db()
        # session_scope() 是同步上下文管理器：正常退出时自动 commit，
        # 抛异常时自动 rollback，无论如何都会 close。批量播种角色属于一次性
        # 写操作，用 contextmanager 而不是 FastAPI 依赖注入更合适
        #（lifespan 不是请求上下文，拿不到 Depends）。
        with session_scope() as db:
            # 幂等播种：已存在则跳过、不存在才插入，保证多次重启不会产生重复角色。
            mapping = persona_service.seed_roles_and_personas(db)
            logger.info("心理医生角色就绪：%s", mapping)
            # 局部导入：只在启动时用一次，避免为整个模块生命周期持有强引用。
            from sqlalchemy import select
            from src.models import SysRole, User, UserSysRole
            # .scalars().first() 取单行 ORM 对象；不写 .scalar() 是为了避免
            # 多列/多行的歧义，语义更明确。
            admin_role = db.execute(select(SysRole).where(SysRole.role_code == "admin")).scalars().first()
            admin_exists = db.execute(
                select(User).where(User.username == settings.admin_username)
            ).scalars().first()
            if not admin_exists and settings.admin_password:
                # 首次部署才会走到这里：用配置里的账号密码创建管理员。
                # 若 admin_password 未配置则跳过，避免创建弱口令账号。
                admin = user_service.register(
                    db, settings.admin_username, settings.admin_password,
                    nickname="系统管理员", is_admin=True,
                )
                logger.info("已创建默认管理员：%s", admin.username)
            elif admin_role:
                # 兼容旧数据/历史版本：用户已存在但可能没有绑定 admin 角色，
                # 这里做一次“补绑”，保证管理员权限不会因升级而丢失。
                exists = db.execute(
                    select(UserSysRole).where(
                        UserSysRole.user_id == admin_exists.id,
                        UserSysRole.role_id == admin_role.id,
                    )
                ).scalars().first()
                if not exists:
                    db.add(UserSysRole(user_id=admin_exists.id, role_id=admin_role.id))
                    # 这里显式 commit：当前用的是 session_scope，
                    # 即使不 commit 外层也会提交；显式写出是为了让“补绑”这一步
                    # 立刻落库可见，便于紧随其后的日志/一致性问题排查。
                    db.commit()
    except Exception as exc:
        # 关键设计：数据库不可用时不中断启动，只记录错误。
        # 理由：Milvus/模型预热仍可完成，服务能以“degraded”状态起来
        #（见 /health 接口），运维可通过健康检查发现问题并修复，
        # 而不是让容器无限重启（CrashLoopBackOff）导致连日志都难看。
        logger.error("MySQL 初始化失败：%s", exc, exc_info=True)

    # 2. Milvus
    try:
        from src.db.milvus import ensure_collections
        # 创建（若不存在）并加载两个 Collection 到内存，
        # 否则首次检索会因 collection 未 load 而报错。
        ensure_collections()
        logger.info("Milvus Collection 就绪：%s", settings.milvus_collection)
    except Exception as exc:
        # 同上，降级而非崩溃：无 Milvus 时系统仍能聊天，只是检索为空。
        logger.error("Milvus 初始化失败：%s", exc, exc_info=True)

    # 3. Redis
    try:
        from src.db.redis import health_check
        # 只做健康探测：Redis 是缓存/短期记忆，连不上也只是丢上下文，
        # 不影响主流程，所以这里不加载数据、不预热 Key。
        logger.info("Redis 健康状态：%s", health_check())
    except Exception as exc:
        logger.error("Redis 初始化失败：%s", exc, exc_info=True)

    # 4. 模型预热（首次调用会加载，预热可降低首问延迟）
    # BGE-M3 与 Reranker 体积大、首次 encode 需数秒到数十秒；
    # 在启动时预热可以把这段时间从“用户第一次提问”转移到“服务启动”。
    import os
    if os.getenv("WARMUP_MODELS", "1") == "1":
        try:
            from src.rag.embedder import get_embedder
            # 用一条极短文本触发模型加载；返回值无意义，仅为了让权重驻留内存。
            get_embedder().encode(["预热"])
            logger.info("BGE-M3 预热完成")
        except Exception as exc:
            # 预热失败不影响正确性（首次真实请求会再次尝试加载），
            # 因此只警告不抛出。
            logger.error("BGE-M3 预热失败（将在首次请求时重试）：%s", exc)
        try:
            from src.rag.reranker import get_reranker
            get_reranker().rerank("预热", [{"text": "预热"}], top_n=1)
            logger.info("BGE-Reranker 预热完成")
        except Exception as exc:
            logger.error("Reranker 预热失败（将在首次请求时重试）：%s", exc)

    logger.info("服务已就绪：http://%s:%s/docs", settings.api_host, settings.api_port)
    # yield 之后进程才算真正开始接收请求；停机时从这里继续执行，
    # 目前无需额外清理（各存储的连接池由进程退出时回收）。
    yield
    logger.info("服务已停止")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    # description 会直接出现在 /docs 页面上，这里放免责声明是为了合规：
    # 让任何拿到接口文档的人第一眼就看到“不做诊断、不开药”的边界。
    description=(
        "基于 RAG 的心理医生多角色陪伴系统。提供三个心理医生角色（人本共情倾听 / CBT / 正念情绪调节），"
        "支持多用户、多角色、多会话、知识库动态更新、短期记忆（Redis）与长期记忆（Milvus）。\n\n"
        "**免责声明**：本系统仅提供心理陪伴、心理教育、自助技巧与就医建议，"
        "不进行医学诊断、不开药、不替代线下医生。"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    # 注意这里必须是显式白名单而不是 ["*"]：
    # allow_credentials=True 与通配符来源在浏览器规范下互斥，
    # 且通配来源会把带 Cookie 的接口暴露给任意站点（CSRF 风险）。
    allow_origins=settings.cors_origins,  # S4：白名单来自 .env 的 CORS_ALLOW_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载业务路由。所有 /api 前缀的接口都在 api_router 里定义。
app.include_router(api_router)

# 聊天前端页面（单文件静态页，访问 /app/）
# 静态目录可缺省：不存在就不挂载，保证纯 API 部署（不含前端产物）也能正常启动。
from pathlib import Path

from fastapi.staticfiles import StaticFiles

# __file__ 是 src/main.py，parent 是 src/，再 parent 才是项目根 => /static。
# 用 resolve() 转成绝对路径，避免 Uvicorn 以其他工作目录启动时找不到文件。
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    # html=True 让访问 /app/ 时自动返回 index.html，省去额外配置。
    app.mount("/app", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """业务异常处理器：把领域异常翻译成统一的 {code, message} 响应。"""
    # 用 warning 而不是 error：这类异常（如参数业务校验、资源不存在）
    # 属于预期内分支，如果打 error 会让告警噪声淹没真正的故障。
    logger.warning("业务异常 %s %s -> %s", request.method, request.url.path, exc.message)
    return JSONResponse(status_code=exc.http_status, content=fail(exc.code, exc.message))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """请求参数校验失败处理器（Pydantic 层）。"""
    logger.warning("参数校验失败 %s：%s", request.url.path, exc.errors())
    # 固定返回 422，并把 Pydantic 的字段级错误列表透传给前端，便于表单高亮。
    return JSONResponse(status_code=422, content=fail(422, "请求参数校验失败", exc.errors()))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    """兜底处理器：捕获所有未被前面处理器覆盖的异常。"""
    # exc_info=True 才会打印完整堆栈，否则只剩一行错误信息，线上无法定位。
    logger.error("未处理异常 %s %s：%s", request.method, request.url.path, exc, exc_info=True)
    # G1：不向客户端泄露内部异常细节，详情只写日志
    # 数据库报错、栈信息等可能包含表结构乃至凭据，泄露给调用方是安全风险。
    return JSONResponse(status_code=500, content=fail(500, "服务器内部错误，请稍后重试或联系管理员"))


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """访问日志中间件：记录耗时与状态码，并注入请求追踪 ID。"""
    # 用单调不保证但足够用的 time.time()：这里只关心相对耗时差。
    start = time.time()
    # 阶段 4：请求级追踪 ID（客户端可传入 X-Request-ID，否则生成），全链路日志可检索
    # 优先复用客户端传入的值：这样跨服务调用时，一次用户操作在网关/本服务/下游
    # 的日志能串成一条链路。截断到 12 位只是为了让日志更短，仍足够唯一。
    rid = request.headers.get("x-request-id") or uuid4().hex[:12]
    set_request_id(rid)
    # call_next 才真正执行后续中间件与路由；异常处理器在外层兜底，
    # 所以这里不需要 try/except（FastAPI 会把异常转成响应再回到这行）。
    response = await call_next(request)
    cost = (time.time() - start) * 1000
    # 跳过文档相关的轮询式访问，避免 /docs 的静态资源把日志刷满。
    if not request.url.path.startswith(("/docs", "/openapi.json", "/redoc")):
        logger.info("%s %s -> %s (%.0fms)", request.method, request.url.path,
                    response.status_code, cost)
    # 把耗时和追踪 ID 回写到响应头，方便前端/压测工具直接观测，无需查日志。
    response.headers["X-Process-Time-Ms"] = f"{cost:.0f}"
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/", tags=["系统"], summary="服务信息")
async def root():
    """根接口：返回服务基本信息，常用于探活与人工确认版本。"""
    return ok({
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "disclaimer": "本系统仅提供心理陪伴与心理教育，不进行诊断、不开药、不替代线下就医。",
    })


@app.get("/health", tags=["系统"], summary="健康检查")
async def health():
    """健康检查：分别探测三个外部依赖，任一失败即标记为 degraded。

    注意返回码仍是 200：这样编排系统（K8s/负载均衡）能拿到详细的 JSON 内容，
    由调用方按 status 字段决定是否摘除节点，而不是只看到一个裸的 503。
    """
    # 局部导入：健康检查不是高频接口，避免为了它在模块加载期就拉起
    # Milvus 客户端 / Redis 连接池。
    from src.db import milvus as milvus_db
    from src.db import redis as redis_db
    from src.db.mysql import health_check as mysql_health

    # 三个检查各自内部已捕获异常并返回 bool，所以这里不会因某个依赖挂掉而 500。
    mysql_ok = mysql_health()
    redis_ok = redis_db.health_check()
    milvus_ok = milvus_db.health_check()
    # 全部就绪才是 ok，否则 degraded（服务仍可用，只是能力降级）。
    status = "ok" if (mysql_ok and redis_ok and milvus_ok) else "degraded"
    return ok({"status": status, "mysql": mysql_ok, "redis": redis_ok, "milvus": milvus_ok})


if __name__ == "__main__":
    # 直接 `python -m src.main` 的本地调试入口；
    # 生产一般用 uvicorn/gunicorn 命令行启动，走的是 `src.main:app` 这条路径。
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        # uvicorn 的日志级别是小写字符串（如 "info"），而配置里通常写大写。
        log_level=settings.log_level.lower(),
    )