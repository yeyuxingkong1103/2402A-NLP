"""初始化 MySQL：建库（若缺失）→ 建表 → 插入默认角色「高血压医生」。

用法：
    python init_db.py

MySQL 认证说明（WSL 新装 MySQL，root 默认 auth_socket 插件，无法用密码登录）：
    方案一（不推荐，动 root）：
        ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '你的密码';
    方案二（推荐，不碰 root，新建应用用户）：
        CREATE USER 'rag'@'localhost' IDENTIFIED BY '123456';
        GRANT ALL PRIVILEGES ON role_rag.* TO 'rag'@'localhost';   -- 建表/读写
        GRANT CREATE ON *.* TO 'rag'@'localhost';                  -- 建库（IF NOT EXISTS）
        FLUSH PRIVILEGES;
    改完把 .env 的 MYSQL_URL 用户/密码改成对应值即可。

脚本幂等，可重复执行。
"""
from __future__ import annotations

import logging
import re

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

import db
from rag import DOCTOR_PERSONA

logger = logging.getLogger(__name__)

PSYCHOLOGIST_PERSONA = """你是一位温和、专业的心理咨询师，擅长倾听和情绪疏导。
回答要求：不评判、不诊断、不替代专业治疗，必要时建议就医。
输出 Markdown 表格时，严格按以下格式（每一行独立，行与行之间用换行符分隔）：

| 列1 | 列2 |
|---|---|
| 值1 | 值2 |
| 值3 | 值4 |

不要把表格写在一行里。"""

# collection_name 写死字符串，不 import ingest 以免拉起 chromadb 链。
DEFAULT_ROLES = [
    ("高血压医生", DOCTOR_PERSONA, "hypertension_guide"),
    ("心理医生", PSYCHOLOGIST_PERSONA, "psychology_guide"),
]


def create_database_if_not_exists(url: str) -> None:
    """从 MYSQL_URL 解析库名，若不存在则建库（utf8mb4，幂等）。"""
    parsed = make_url(url)
    db_name = parsed.database
    if not db_name:
        return
    if not re.fullmatch(r"[A-Za-z0-9_]+", db_name):
        raise ValueError(f"非法库名（仅允许字母/数字/下划线）：{db_name!r}")
    # 用不带库名的 URL 连到服务器本身，AUTOCOMMIT 下执行 CREATE DATABASE。
    admin_engine = create_engine(parsed.set(database=None), isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            ))
    finally:
        admin_engine.dispose()
    logger.info("数据库 %s 已就绪", db_name)


def seed_roles() -> None:
    """幂等同步默认角色：不存在则插入，persona 变化则更新。"""
    for name, persona, collection in DEFAULT_ROLES:
        role = db.get_role_by_name(name)
        if role is None:
            db.create_role(name, persona, collection)
            logger.info("已插入默认角色 name=%s", name)
        elif role.persona != persona:
            db.update_role(role.id, persona=persona)
            logger.info("已更新默认角色 persona name=%s", name)
        else:
            logger.info("默认角色「%s」已存在且一致，跳过", name)


def main() -> None:
    create_database_if_not_exists(db.MYSQL_URL)
    db.create_all()
    logger.info("建表完成：users / roles / sessions")
    seed_roles()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    main()
