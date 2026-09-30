"""
init_db.py — 初始化数据库与 Milvus 集合

    python scripts/init_db.py

依次完成：
    1. 创建 Milvus 集合：kb_legal / kb_medical / kb_english / long_term_memory
    2. 创建 MySQL 库表：documents（知识库文档元数据）、conversations（对话记录）
    3. 探测 Redis 连通性

所有步骤都是幂等的，可以重复执行。
LOCAL_MODE=True 时跳过 Milvus 与 Redis，只做本地目录和 SQLite 无关的检查，
不需要任何外部中间件。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

DDL_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS documents (
        id INT AUTO_INCREMENT PRIMARY KEY,
        domain VARCHAR(32) NOT NULL COMMENT '所属领域',
        filename VARCHAR(255) NOT NULL COMMENT '文件名',
        pages INT DEFAULT 0 COMMENT '页数',
        chunks INT DEFAULT 0 COMMENT '入库分块数',
        created_at INT COMMENT '创建时间戳',
        updated_at INT COMMENT '更新时间戳',
        UNIQUE KEY uk_domain_filename (domain, filename)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库文档元数据'""",
    """CREATE TABLE IF NOT EXISTS conversations (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(128) NOT NULL COMMENT '用户标识',
        domain VARCHAR(32) COMMENT '识别到的领域',
        role VARCHAR(64) COMMENT '应答角色',
        question TEXT COMMENT '用户问题',
        answer MEDIUMTEXT COMMENT '模型回答',
        sources TEXT COMMENT '引用的资料来源 JSON',
        latency FLOAT COMMENT '耗时秒数',
        created_at INT COMMENT '创建时间戳',
        KEY idx_user_time (user_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话记录'""",
    # 用户信息由 users.py 读写，列定义必须与 users.py 里的 _MYSQL_SCHEMA 一致
    """CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(32) UNIQUE NOT NULL,
        created_at BIGINT NOT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户信息'""",
    # 角色信息对应 domains/list_roles() 的四个字段
    """CREATE TABLE IF NOT EXISTS roles (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(64) NOT NULL COMMENT '角色名称',
        domain VARCHAR(32) NOT NULL COMMENT '所属领域',
        identity VARCHAR(255) COMMENT '角色身份说明',
        greeting VARCHAR(255) COMMENT '开场白',
        created_at INT COMMENT '创建时间戳',
        updated_at INT COMMENT '更新时间戳',
        UNIQUE KEY uk_domain_name (domain, name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='角色信息'""",
]


def init_milvus() -> bool:
    """创建所有需要的 Milvus 集合。"""
    print("[1/3] 初始化 Milvus 集合")

    if config.LOCAL_MODE:
        print("      跳过：LOCAL_MODE=True，当前使用本地向量库")
        return True

    try:
        from milvus_store import MilvusStore

        store = MilvusStore()
    except Exception as exc:
        print(f"      失败：无法连接 Milvus（{exc}）")
        print(f"      请确认 MILVUS_URI={config.MILVUS_URI} 可访问")
        return False

    collections = list(config.KB_COLLECTIONS.values()) + [config.LONG_TERM_COLLECTION]
    for collection in collections:
        try:
            existed = store.client.has_collection(collection)
            store.ensure_collection(collection)
            print(f"      {'已存在' if existed else '已创建'}  {collection}")
        except Exception as exc:
            print(f"      失败：创建 {collection} 出错（{exc}）")
            return False

    store.close()
    return True


def init_mysql() -> bool:
    """创建 MySQL 库与表。连不上时给出提示但不视为致命错误。"""
    print("[2/3] 初始化 MySQL")

    if config.LOCAL_MODE:
        print("      跳过：LOCAL_MODE=True，本轮对话记录不落 MySQL")
        return True

    try:
        import pymysql
    except ImportError:
        print("      跳过：未安装 pymysql（pip install pymysql）")
        return True

    try:
        connection = pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            charset="utf8mb4",
            connect_timeout=5,
        )
    except Exception as exc:
        print(f"      失败：无法连接 MySQL（{exc}）")
        print("      请确认 MySQL 已启动且 config.py 中的账号密码正确")
        return False

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DB}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE `{config.MYSQL_DB}`")
            for ddl in DDL_STATEMENTS:
                cursor.execute(ddl)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
        print(f"      数据库 {config.MYSQL_DB} 就绪，表：{', '.join(tables)}")
    finally:
        connection.close()
    return True


def check_redis() -> bool:
    """探测 Redis 连通性。"""
    print("[3/3] 检查 Redis")

    if config.LOCAL_MODE:
        print("      跳过：LOCAL_MODE=True，短期记忆写入本地文件")
        return True

    try:
        import redis

        client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            password=config.REDIS_PASSWORD or None,
            socket_connect_timeout=3,
        )
        client.ping()
        print(f"      连通正常 {config.REDIS_HOST}:{config.REDIS_PORT}")
        return True
    except Exception as exc:
        print(f"      失败：{exc}")
        return False


def main() -> int:
    config.ensure_dirs()
    print(f"运行模式：{'本地轻量 (LOCAL_MODE=True)' if config.LOCAL_MODE else '生产'}")

    results = [init_milvus(), init_mysql(), check_redis()]

    print()
    if all(results):
        print("初始化完成。")
        return 0

    print("初始化过程中有步骤失败，请根据上面的提示处理后重试。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
