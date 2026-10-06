"""users 表结构对齐迁移（批次5 认证落库）。

目标结构：email(唯一) / password_hash / is_admin / is_active / created_at / updated_at
         + user_key(32位十六进制，唯一)
变更（users 当前 0 行，无数据损失）：
  1. ADD COLUMN email VARCHAR(255) NOT NULL + 唯一键 uk_users_email
  2. ADD COLUMN user_key VARCHAR(32) NOT NULL + 唯一键 uk_users_user_key
  3. ADD COLUMN is_admin TINYINT(1) NOT NULL DEFAULT 0
  4. DROP COLUMN username（唯一索引 username 随列删除）
幂等：每步先查 information_schema，列/索引已存在则跳过并打印 SKIP。
红线：不 DROP 表、不动 users 之外任何表；chat_sessions 的 7 条孤儿会话保留不动。
"""
import sys
from pathlib import Path

# 项目根（scripts/migrations → scripts → rag），供下面导入 scripts/_env 使用
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pymysql

from scripts._env import load_project_env, mysql_config

# 连接参数从 .env 读（MYSQL_HOST/PORT/USER/PASSWORD/DATABASE），顺带剔除 http(s)_proxy；
# 源码里不再出现任何口令字面量
load_project_env(PROJECT_ROOT)
_CFG = mysql_config()
HOST, PORT, USER, PASSWORD, DATABASE = (
    _CFG["host"],
    _CFG["port"],
    _CFG["user"],
    _CFG["password"],
    _CFG["database"],
)


def column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_NAME=%s",
        (DATABASE, table, column),
    )
    return cur.fetchone()[0] > 0


def index_exists(cur, table: str, index: str) -> bool:
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND INDEX_NAME=%s",
        (DATABASE, table, index),
    )
    return cur.fetchone()[0] > 0


def row_count(cur, table: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]


def show_structure(cur) -> None:
    cur.execute("SHOW CREATE TABLE users")
    print(cur.fetchone()[1])


def main() -> None:
    conn = pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE,
        charset="utf8mb4",
    )
    cur = conn.cursor()

    print("== 迁移前结构 ==")
    show_structure(cur)
    print("users 行数:", row_count(cur, "users"))

    assert row_count(cur, "users") == 0, "users 表非空，禁止自动迁移，需人工评估"
    assert not column_exists(cur, "users", "email"), "email 列已存在，可能已迁移过"

    # 1. email 列 + 唯一键
    if not column_exists(cur, "users", "email"):
        cur.execute(
            "ALTER TABLE users ADD COLUMN email VARCHAR(255) NOT NULL AFTER id"
        )
        print("[OK] ADD COLUMN email")
    if not index_exists(cur, "users", "uk_users_email"):
        cur.execute(
            "ALTER TABLE users ADD UNIQUE KEY uk_users_email (email)"
        )
        print("[OK] ADD UNIQUE KEY uk_users_email")

    # 2. user_key 列 + 唯一键
    if not column_exists(cur, "users", "user_key"):
        cur.execute(
            "ALTER TABLE users ADD COLUMN user_key VARCHAR(32) NOT NULL AFTER id"
        )
        print("[OK] ADD COLUMN user_key")
    if not index_exists(cur, "users", "uk_users_user_key"):
        cur.execute(
            "ALTER TABLE users ADD UNIQUE KEY uk_users_user_key (user_key)"
        )
        print("[OK] ADD UNIQUE KEY uk_users_user_key")

    # 3. is_admin 列
    if not column_exists(cur, "users", "is_admin"):
        cur.execute(
            "ALTER TABLE users ADD COLUMN is_admin TINYINT(1) NOT NULL DEFAULT 0 "
            "AFTER password_hash"
        )
        print("[OK] ADD COLUMN is_admin")

    # 4. 删除 username 列（唯一索引随列自动删除）
    if column_exists(cur, "users", "username"):
        cur.execute("ALTER TABLE users DROP COLUMN username")
        print("[OK] DROP COLUMN username（唯一索引 username 随列删除）")

    conn.commit()

    print("== 迁移后结构 ==")
    show_structure(cur)
    print("users 行数:", row_count(cur, "users"))
    cur.execute("SELECT COUNT(*) FROM chat_sessions")
    print("chat_sessions 行数（未动）:", cur.fetchone()[0])
    conn.close()
    print("MIGRATION_DONE")


if __name__ == "__main__":
    main()
