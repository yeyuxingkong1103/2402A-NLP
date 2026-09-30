"""把旧 SQLite 用户导入 MySQL；不删除原数据库。"""
import argparse
import sqlite3

from app import db


def migrate(path):
    if not path.exists():
        raise FileNotFoundError(path)
    source = sqlite3.connect(path)
    source.row_factory = sqlite3.Row
    users = source.execute("SELECT username, password_hash, created_at FROM users").fetchall()
    source.close()
    db.init_db()
    inserted = 0
    with db.transaction() as target:
        for row in users:
            cursor = db.execute(target,
                "INSERT IGNORE INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                (row["username"], row["password_hash"], row["created_at"]))
            inserted += cursor.rowcount
            cursor.close()
    print(f"MYSQL_MIGRATION_OK: 读取 {len(users)}，新增 {inserted}；原 SQLite 未删除")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=__import__("pathlib").Path, nargs="?", default="db/app.db")
    migrate(parser.parse_args().path)
