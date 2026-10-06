from datetime import UTC, datetime

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """所有 SQLAlchemy ORM 模型共用的 declarative 根类。"""

    pass


def utc_now() -> datetime:
    """统一生成带时区的 UTC 时间，避免数据库中混入本地时间。"""
    return datetime.now(UTC)
