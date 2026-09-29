# scripts/init_db.py
"""初始化 MySQL：建库、建表、写入角色种子数据，并创建一个默认用户。

用法:
    .venv/bin/python -m scripts.init_db
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from app.config import settings
from app.core.role_service import RoleService
from app.db import mysql_conn
from app.models.tables import User
from app.config.logging_conf import setup_logging
import logging

logger = logging.getLogger(__name__)

setup_logging()


def create_default_user() -> None:
    """建一个演示用户，方便直接调 /api/chat/ask（user_id=1）。"""
    from app.api.system import hash_password
    with mysql_conn.session_scope() as db:
        exists = db.execute(select(User).where(
            User.username == "demo")).scalars().first()
        if exists:
            logger.info("默认用户 demo 已存在 (id=%s)", exists.id)
            return
        u = User(username="demo", password_hash=hash_password("demo123"),
                 nickname="演示用户")
        db.add(u)
        db.flush()
        logger.info("已创建默认用户 demo / demo123 (id=%s)", u.id)


def main():
    logger.info("目标 MySQL: %s:%s/%s", settings.MYSQL_HOST,
                settings.MYSQL_PORT, settings.MYSQL_DATABASE)

    mysql_conn.create_database_if_missing()
    mysql_conn.init_tables()

    with mysql_conn.session_scope() as db:
        added = RoleService.seed_defaults(db)

    with mysql_conn.session_scope() as db:
        roles = RoleService.list_roles(db, only_active=False)
        logger.info("当前角色 %s 个:", len(roles))
        for r in roles:
            logger.info("  [%s] %s —— %s", r.role_key, r.name, r.category)

    create_default_user()
    logger.info("数据库初始化完成（新增角色 %s 个）", added)


if __name__ == "__main__":
    main()
