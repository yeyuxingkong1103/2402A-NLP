"""初始化数据库：建库 + 建表 + 系统角色 + 三个心理医生角色 + 默认管理员。

用法：
    python scripts/init_db.py [--drop-all]
"""
import argparse
import os
import sys

# 把项目根目录插入 sys.path 首位，使脚本能以绝对路径方式被任意位置调用时
# 仍能 import 到 src.* 包（否则脱离项目根运行时找不到模块）。放在 import 之前。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings  # noqa: E402
from src.core.logging import get_logger, setup_logging  # noqa: E402
from src.db.mysql import get_engine, init_db, session_scope  # noqa: E402
from src.db.milvus import ensure_collections  # noqa: E402
from src.models import Base  # noqa: E402
from src.services import persona_service, user_service  # noqa: E402

logger = get_logger("scripts.init_db")


def main() -> None:
    # argparse 解析命令行参数，让脚本既适合交互执行也适合在 install.sh 中被调用
    parser = argparse.ArgumentParser(description="初始化 MySQL 与 Milvus")
    parser.add_argument("--drop-all", action="store_true", help="删除并重建所有表（危险）")
    parser.add_argument("--skip-milvus", action="store_true", help="跳过 Milvus 初始化")
    args = parser.parse_args()

    setup_logging()
    logger.info("开始初始化数据库：%s:%s/%s", settings.db_host, settings.db_port, settings.db_name)

    if args.drop_all:
        # 危险操作：先确保数据库存在，再 drop 全部表（由 SQLAlchemy metadata 反射得到）
        from src.db.mysql import create_database_if_not_exists
        create_database_if_not_exists()
        Base.metadata.drop_all(bind=get_engine())
        logger.warning("已删除全部数据表")

    # 建库建表（若已存在则跳过，幂等）
    init_db()

    with session_scope() as db:
        # 写入系统角色与三个心理医生角色（seed 函数内部按 code 幂等，重复执行不产生重复数据）
        mapping = persona_service.seed_roles_and_personas(db)
        logger.info("角色初始化：%s", mapping)

        from sqlalchemy import select
        from src.models import User
        # 管理员也按用户名查重：已存在则跳过，避免重复注册报唯一约束冲突
        admin = db.execute(
            select(User).where(User.username == settings.admin_username)
        ).scalars().first()
        if not admin:
            admin = user_service.register(
                db, settings.admin_username, settings.admin_password,
                nickname="系统管理员", is_admin=True,
            )
            logger.info("默认管理员已创建：%s / %s", settings.admin_username, settings.admin_password)
        else:
            logger.info("管理员已存在：%s", settings.admin_username)

    # 默认初始化 Milvus 向量集合；--skip-milvus 用于仅初始化 MySQL 的场景
    if not args.skip_milvus:
        ensure_collections()
        logger.info("Milvus Collection 初始化完成")

    logger.info("初始化完成 ✅")


if __name__ == "__main__":
    main()