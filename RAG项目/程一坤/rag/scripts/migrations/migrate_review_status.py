"""阶段6 审核状态迁移（显式、幂等）。

变更：
  1. document_versions 加可追溯列：reviewed_by / reviewed_at / review_note（审核动作留痕）
  2. 存量 11 个版本 version_status 'new' -> 'approved'

  为什么迁移：这批版本已在 Milvus 建索引且在正常使用（11/11 indexed），
  它们是审核机制上线前导入的存量内容，等价于"已通过人工审核并发布"。
  不迁移的话，检索侧兜底过滤（只读 approved）会让这 11 篇法规全部消失。
  迁移是显式 UPDATE，只动 version_status='new' 的行，打印影响清单。

红线：不 DROP 表、不改其它列、不动其它表。
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
        "WHERE TABLE_SCHEMA=%s AND TABLE_NAME='document_versions' AND COLUMN_NAME=%s",
        (DATABASE, column),
    )
    return cur.fetchone()[0] > 0


def main() -> None:
    conn = pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE,
        charset="utf8mb4",
    )
    cur = conn.cursor()

    cur.execute("SELECT version_status, COUNT(*) FROM document_versions GROUP BY version_status")
    print("迁移前 version_status 分布:", cur.fetchall())

    # 1. 可追溯列（幂等）
    for ddl, name in [
        ("ALTER TABLE document_versions ADD COLUMN reviewed_by VARCHAR(32) NULL", "reviewed_by"),
        ("ALTER TABLE document_versions ADD COLUMN reviewed_at DATETIME NULL", "reviewed_at"),
        ("ALTER TABLE document_versions ADD COLUMN review_note VARCHAR(1024) NULL", "review_note"),
    ]:
        if not column_exists(cur, name):
            cur.execute(ddl)
            print(f"[OK] ADD COLUMN {name}")
        else:
            print(f"[SKIP] {name} 已存在")

    # 2. 存量 new -> approved（显式列出将影响的行）
    cur.execute(
        "SELECT id, version_key, created_at FROM document_versions "
        "WHERE version_status='new' ORDER BY id"
    )
    affected = cur.fetchall()
    print(f"将迁移为 approved 的版本共 {len(affected)} 条:")
    for row in affected:
        print("  ", row)
    cur.execute(
        "UPDATE document_versions SET version_status='approved' WHERE version_status='new'"
    )
    print(f"[OK] UPDATE version_status new->approved，影响 {cur.rowcount} 行")
    conn.commit()

    cur.execute("SELECT version_status, COUNT(*) FROM document_versions GROUP BY version_status")
    print("迁移后 version_status 分布:", cur.fetchall())
    conn.close()
    print("REVIEW_STATUS_MIGRATION_DONE")


if __name__ == "__main__":
    main()
