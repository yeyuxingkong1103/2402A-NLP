import logging
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import text

from backend.app.core.config import AppSettings
from backend.app.database.milvus import check_milvus_readiness
from backend.app.workers.celery_app import check_celery_readiness

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    ok: bool
    detail: str | None = None


def is_ready(statuses: Iterable[DependencyStatus]) -> bool:
    # 所有依赖均通过时才允许 ready，空列表按 Python all 语义视为通过。
    return all(status.ok for status in statuses)


def _check_mysql_readiness(settings: AppSettings) -> DependencyStatus:
    if not settings.DATABASE_URL:
        return DependencyStatus("mysql", False, "missing database url")
    try:
        from backend.app.repositories.auth_repository import create_engine_from_url

        engine = create_engine_from_url(settings.DATABASE_URL, connect_args={"connect_timeout": 2})
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        logger.warning("MySQL readiness check failed", extra={"error_type": type(exc).__name__})
        return DependencyStatus("mysql", False, type(exc).__name__)
    return DependencyStatus("mysql", True)


def _check_redis_readiness(settings: AppSettings) -> DependencyStatus:
    if not settings.REDIS_URL:
        return DependencyStatus("redis", False, "missing redis url")
    try:
        import redis

        client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        client.close()
    except Exception as exc:
        logger.warning("Redis readiness check failed", extra={"error_type": type(exc).__name__})
        return DependencyStatus("redis", False, type(exc).__name__)
    return DependencyStatus("redis", True)


def _check_required_value(name: str, value: str, missing_detail: str) -> DependencyStatus:
    # 仅检查值是否存在，不记录真实配置内容，避免泄露密钥或敏感路径细节。
    has_value = bool(value)
    # 日志只包含依赖名和脱敏状态，便于排查启动问题。
    logger.info("startup dependency checked", extra={"dependency": name, "ok": has_value})
    # 缺失时返回固定原因，调用方可直接展示给健康检查消费者。
    return DependencyStatus(name=name, ok=has_value, detail=None if has_value else missing_detail)


def run_startup_checks(settings: AppSettings) -> list[DependencyStatus]:
    # 生产 readiness 只做脱敏状态汇总，不返回连接串、密钥、模型路径或用户文本。
    statuses = [
        _check_mysql_readiness(settings),
        _check_redis_readiness(settings),
    ]
    milvus_ok, milvus_detail = check_milvus_readiness(settings)
    statuses.append(DependencyStatus("milvus", milvus_ok, milvus_detail))
    celery_ok, celery_detail = check_celery_readiness()
    statuses.append(DependencyStatus("celery", celery_ok, celery_detail))
    # DeepSeek 仅检查密钥是否配置，绝不输出密钥值。
    statuses.append(_check_required_value("deepseek", settings.DEEPSEEK_API_KEY, "missing api key"))
    # 模型 readiness 显式加载本地 CUDA 模型，失败只返回异常类型。
    from backend.app.embeddings.embedding_factory import check_model_readiness

    statuses.extend(check_model_readiness())
    # 汇总日志只输出数量和状态，不包含密钥、手机号或聊天原文。
    logger.info("startup checks completed", extra={"dependency_count": len(statuses), "ready": is_ready(statuses)})
    return statuses
