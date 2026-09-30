import logging

from backend.app.core.config import AppSettings, settings
from backend.app.repositories.auth_repository import SQLAlchemyAuthStore, create_engine_from_url
from backend.app.repositories.otp_repository import RedisOtpStore
from backend.app.services.auth_service import get_auth_store, reset_auth_store, set_auth_store

logger = logging.getLogger(__name__)


class AuthStoreConfigurationError(RuntimeError):
    # 启动期配置错误使用专用异常，避免吞掉生产存储未配置问题。
    pass


def configure_auth_store(app_settings: AppSettings = settings) -> None:
    # 显式根据配置选择认证和聊天存储，默认仍为内存以保持本地测试和 MVP 行为稳定。
    backend = app_settings.AUTH_STORE_BACKEND.strip().lower()
    if backend == "memory":
        reset_auth_store()
        _configure_chat_repository(None)
        _configure_memory_repositories(None)
        _configure_feedback_repository(None)
    elif backend == "sql":
        _configure_sql_auth_store(app_settings)
    else:
        logger.error("unsupported auth store backend", extra={"backend": backend})
        raise AuthStoreConfigurationError("unsupported auth store backend")
    _configure_request_controller(app_settings)
    _configure_otp_store(app_settings)
    logger.info("auth store configured", extra={"backend": backend, "otp_backend": app_settings.OTP_STORE_BACKEND.strip().lower(), "request_control_backend": app_settings.REQUEST_CONTROL_BACKEND.strip().lower()})


def _configure_sql_auth_store(app_settings: AppSettings) -> None:
    # SQL 模式必须显式提供数据库连接串；日志不输出连接串以免泄露凭据。
    if not app_settings.DATABASE_URL:
        logger.error("sql auth store missing database url")
        raise AuthStoreConfigurationError("DATABASE_URL is required for sql auth store")
    engine = create_engine_from_url(app_settings.DATABASE_URL)
    store = SQLAlchemyAuthStore(engine)
    set_auth_store(store)
    _configure_chat_repository(engine)
    _configure_memory_repositories(engine)
    _configure_feedback_repository(engine)


def _configure_chat_repository(engine) -> None:
    # 延迟导入 API 服务，避免存储配置层与路由初始化形成循环依赖。
    from backend.app.api.v1.chat import chat_service

    if engine is None:
        chat_service.repository = None
        return
    from backend.app.repositories.chat_repository import SQLAlchemyChatRepository

    chat_service.repository = SQLAlchemyChatRepository(engine)


def _configure_memory_repositories(engine) -> None:
    # 记忆和导出服务共享同一个 SQL repository；默认内存模式显式清空注入。
    from backend.app.services import export_service, memory_service

    if engine is None:
        memory_service.set_memory_repository(None)
        export_service.set_export_repository(None)
        return
    from backend.app.repositories.memory_repository import SQLAlchemyMemoryRepository

    repository = SQLAlchemyMemoryRepository(engine)
    memory_service.set_memory_repository(repository)
    export_service.set_export_repository(repository)


def _configure_feedback_repository(engine) -> None:
    # 反馈和高风险告警在 SQL 模式下持久化；默认内存模式保持原测试行为。
    from backend.app.services import feedback_service

    if engine is None:
        feedback_service.set_feedback_repository(None)
        return
    from backend.app.repositories.feedback_repository import SQLAlchemyFeedbackRepository

    feedback_service.set_feedback_repository(SQLAlchemyFeedbackRepository(engine))


def _configure_request_controller(app_settings: AppSettings) -> None:
    # 请求控制可切换 Redis 后端，使限流和并发占用在多 worker 间共享。
    from backend.app.api.v1.chat import chat_service
    from backend.app.services.request_control import create_request_controller

    try:
        chat_service._request_controller = create_request_controller(app_settings)
    except ValueError as exc:
        logger.error("request control configuration failed", extra={"backend": app_settings.REQUEST_CONTROL_BACKEND.strip().lower()})
        raise AuthStoreConfigurationError(str(exc)) from exc


def _configure_otp_store(app_settings: AppSettings) -> None:
    # OTP 只有在明确配置 Redis 时才跨进程存储，避免默认开发环境依赖外部服务。
    backend = app_settings.OTP_STORE_BACKEND.strip().lower()
    if backend == "memory":
        get_auth_store().set_otp_backend(None)
        return
    if backend != "redis":
        logger.error("unsupported otp store backend", extra={"backend": backend})
        raise AuthStoreConfigurationError("unsupported OTP_STORE_BACKEND")
    if not app_settings.REDIS_URL:
        logger.error("redis otp store missing redis url")
        raise AuthStoreConfigurationError("REDIS_URL is required for redis OTP store")
    import redis

    client = redis.from_url(app_settings.REDIS_URL, decode_responses=True)
    get_auth_store().set_otp_backend(RedisOtpStore(client))
