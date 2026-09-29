"""批次 14 迁移：users 表加长期记忆开关列（显式、幂等）。

变更：
  users 加列 long_term_memory_enabled BOOLEAN NOT NULL DEFAULT TRUE
  （接口文档 8.3 PUT /api/v1/users/me/memory-settings 的落库字段；存量用户默认开启，
  与"用户可关闭长期记忆"语义一致——关闭是主动行为）

红线：不 DROP、不改其它列、不动其它表。
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


def column_exists(cur, column: str) -> bool:
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=%s AND TABLE_NAME='users' AND COLUMN_NAME=%s",
        (DATABASE, column),
    )
    return cur.fetchone()[0] > 0


def main() -> None:
    conn = pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE,
        charset="utf8mb4",
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    print("迁移前用户数:", cur.fetchone()[0])

    if not column_exists(cur, "long_term_memory_enabled"):
        cur.execute(
            "ALTER TABLE users ADD COLUMN long_term_memory_enabled "
            "BOOLEAN NOT NULL DEFAULT TRUE"
        )
        conn.commit()
        print("已加列 users.long_term_memory_enabled")
    else:
        print("列已存在，跳过")

    cur.execute(
        "SELECT long_term_memory_enabled, COUNT(*) FROM users GROUP BY long_term_memory_enabled"
    )
    print("迁移后开关分布:", cur.fetchall())
    conn.close()


if __name__ == "__main__":
    main()
