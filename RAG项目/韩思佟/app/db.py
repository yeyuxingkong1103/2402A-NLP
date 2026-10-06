"""用户与医生角色存 MySQL；只有显式测试模式使用 SQLite。"""
import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager

from app.config import BASE, enabled, setting
from app.prompts import GENERAL_DOCTOR_PROMPT

DB_PATH = BASE / "db" / "app.db"
DOCTOR = ("医生", GENERAL_DOCTOR_PROMPT,
          "通用医学健康科普医生（优先引用高血压指南知识库）")


def backend():
    return "sqlite" if enabled("RAG_TEST_MODE") else setting("RAG_DB_BACKEND", "mysql").lower()


def get_conn():
    if backend() == "sqlite":
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    if backend() != "mysql":
        raise RuntimeError("RAG_DB_BACKEND must be mysql or sqlite")
    import mysql.connector
    return mysql.connector.connect(
        host=setting("RAG_MYSQL_HOST", "127.0.0.1"),
        port=int(setting("RAG_MYSQL_PORT", "3306")),
        user=setting("RAG_MYSQL_USER", "rag_user"),
        password=setting("RAG_MYSQL_PASSWORD", "rag_password"),
        database=setting("RAG_MYSQL_DATABASE", "rag_roleplay"),
        charset="utf8mb4", connection_timeout=5,
    )


@contextmanager
def transaction():
    """成功提交，异常回滚，始终关闭连接；SQL值使用参数绑定。"""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute(conn, sql, params=()):
    sqlite_mode = isinstance(conn, sqlite3.Connection)
    cur = conn.cursor() if sqlite_mode else conn.cursor(dictionary=True)
    if sqlite_mode:
        sql = sql.replace("INSERT IGNORE", "INSERT OR IGNORE")
    else:
        sql = sql.replace("?", "%s")
    cur.execute(sql, params)
    return cur


def init_db():
    with transaction() as conn:
        sqlite_mode = isinstance(conn, sqlite3.Connection)
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if sqlite_mode else "BIGINT PRIMARY KEY AUTO_INCREMENT"
        suffix = "" if sqlite_mode else " CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
        for sql in (
            f"""CREATE TABLE IF NOT EXISTS users (
                id {pk}, username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP){suffix}""",
            f"""CREATE TABLE IF NOT EXISTS roles (
                id {pk}, name VARCHAR(50) UNIQUE NOT NULL,
                persona_prompt TEXT NOT NULL, description VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP){suffix}""",
        ):
            execute(conn, sql).close()
        verb = "INSERT OR IGNORE" if sqlite_mode else "INSERT IGNORE"
        execute(conn, f"{verb} INTO roles(name, persona_prompt, description) VALUES (?, ?, ?)", DOCTOR).close()
        execute(conn, "UPDATE roles SET persona_prompt = ?, description = ? WHERE name = ?",
                (DOCTOR[1], DOCTOR[2], DOCTOR[0])).close()


def hash_password(password):
    salt = secrets.token_hex(16)
    rounds = 260_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), rounds)
    return f"pbkdf2_sha256${rounds}${salt}${digest.hex()}"


def password_matches(password, stored):
    try:
        if stored.startswith("pbkdf2_sha256$"):
            _, rounds, salt, expected = stored.split("$", 3)
            rounds = int(rounds)
            if not 100_000 <= rounds <= 2_000_000:
                return False
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), rounds)
            return hmac.compare_digest(digest.hex(), expected)
        # 旧SQLite账号迁移后，第一次成功登录时升级口令格式。
        return hmac.compare_digest(hashlib.sha256(password.encode()).hexdigest(), stored)
    except (ValueError, TypeError):
        return False


def _insert(sql, values):
    try:
        with transaction() as conn:
            cur = execute(conn, sql, values)
            uid = cur.lastrowid
            cur.close()
            return uid
    except Exception as exc:
        if isinstance(exc, sqlite3.IntegrityError) or getattr(exc, "errno", None) == 1062:
            return None
        raise


def create_user(username, password):
    return _insert("INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                   (username, hash_password(password)))


def verify_user(username, password):
    with transaction() as conn:
        cur = execute(conn, "SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        cur.close()
        if not row or not password_matches(password, row["password_hash"]):
            return None
        if not row["password_hash"].startswith("pbkdf2_sha256$"):
            execute(conn, "UPDATE users SET password_hash = ? WHERE id = ?",
                    (hash_password(password), row["id"])).close()
        return dict(row)


def list_roles():
    with transaction() as conn:
        cur = execute(conn, "SELECT id, name, description FROM roles ORDER BY id")
        rows = [dict(row) for row in cur.fetchall()]
        cur.close()
        return rows


def get_role(role_id):
    with transaction() as conn:
        cur = execute(conn, "SELECT * FROM roles WHERE id = ?", (role_id,))
        row = cur.fetchone()
        cur.close()
        return dict(row) if row else None


def add_role(name, persona_prompt, description=""):
    """仅迁移/测试使用；网页不提供自定义角色入口。"""
    return _insert("INSERT INTO roles(name, persona_prompt, description) VALUES (?, ?, ?)",
                   (name, persona_prompt, description))


def ping():
    with transaction() as conn:
        cur = execute(conn, "SELECT 1")
        cur.fetchone()
        cur.close()
    return True
