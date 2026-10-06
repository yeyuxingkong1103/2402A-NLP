from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_database_engine(database_url: str) -> Engine:
    """按显式连接串创建 Engine；调用方负责避免打印敏感连接信息。"""
    if not database_url:
        raise ValueError("database_url is required")
    return create_engine(database_url)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """创建禁用提交后过期的 Session 工厂，便于 CLI 输出导入结果。"""
    return sessionmaker(bind=engine, expire_on_commit=False)
