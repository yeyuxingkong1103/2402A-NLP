# app/db/mysql_conn.py
"""MySQL 连接：SQLAlchemy 引擎 / Session 工厂 / 建库建表。

承担两类持久化数据：
  - 用户信息（users）
  - 角色信息（roles）
  - 会话与消息归档（sessions / messages），Redis 只保留热数据
"""
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
import logging

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_factory: Optional[sessionmaker] = None
_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = create_engine(
                    settings.mysql_url(),
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,       # 断线自动重连
                    pool_recycle=3600,
                    echo=False,
                    future=True,
                )
                logger.info("MySQL 已连接: %s:%s/%s", settings.MYSQL_HOST,
                            settings.MYSQL_PORT, settings.MYSQL_DATABASE)
    return _engine


def get_session_factory() -> sessionmaker:
    global _factory
    if _factory is None:
        _factory = sessionmaker(bind=get_engine(), autoflush=False,
                                autocommit=False, expire_on_commit=False,
                                future=True)
    return _factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务作用域：正常提交，异常回滚。脚本里用它。"""
    s = get_session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    """FastAPI 依赖注入用的 Session：请求正常结束即提交，异常回滚。

    注意：这里必须显式 commit。只在接口里 flush() 的话，ID 虽然分配了，
    事务却会在 Session 关闭时被回滚——接口返回 200，数据却没落库。
    """
    s = get_session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def create_database_if_missing() -> None:
    """连接到 MySQL 实例（不指定库）并建库。"""
    engine = create_engine(settings.mysql_url(with_database=False),
                           future=True, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE DATABASE IF NOT EXISTS `%s` "
            "DEFAULT CHARACTER SET %s COLLATE %s_unicode_ci"
            % (settings.MYSQL_DATABASE, settings.MYSQL_CHARSET,
               settings.MYSQL_CHARSET)))
    engine.dispose()
    logger.info("数据库 %s 已就绪", settings.MYSQL_DATABASE)


def init_tables() -> None:
    """按 ORM 定义建表（已存在则跳过）。"""
    from app.models.tables import Base
    Base.metadata.create_all(get_engine())
    logger.info("数据表已就绪: %s", ", ".join(Base.metadata.tables.keys()))


def init_database() -> None:
    """建库 + 建表，一步到位。"""
    create_database_if_missing()
    init_tables()


def healthcheck() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:                                  # pragma: no cover
        logger.error("MySQL 健康检查失败: %s", e)
        return False
