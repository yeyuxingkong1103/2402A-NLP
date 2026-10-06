from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Any, Iterator
from urllib.parse import unquote, urlparse
from uuid import uuid4

from ..config import Settings


class MySQLStore:
    """MySQL 连接与基础 schema 管理，业务仓储类复用这里的事务封装。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.mysql_url:
            raise RuntimeError("MYSQL_URL 未配置")
        self.init_schema()

    def params(self) -> dict[str, Any]:
        from pymysql.cursors import DictCursor

        parsed = urlparse(self.settings.mysql_url.replace("mysql+pymysql://", "mysql://", 1))
        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 3306,
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "database": parsed.path.lstrip("/") or None,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "autocommit": False,
        }

    def connect(self):
        import pymysql

        return pymysql.connect(**self.params())

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """为单次数据库操作提供提交/回滚边界。"""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self.transaction() as conn:
            with conn.cursor() as cur:
                return cur.execute(sql, params)

    def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]

    def init_schema(self) -> None:
        """创建当前版本需要的表，并补齐早期版本遗留 users 表。"""
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              user_id VARCHAR(64) PRIMARY KEY,
              username VARCHAR(128) UNIQUE NOT NULL,
              email VARCHAR(255) UNIQUE NOT NULL,
              password_hash VARCHAR(255) NOT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.migrate_legacy_users_table()
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_files (
              document_id VARCHAR(64) PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              session_id VARCHAR(128),
              file_name VARCHAR(255) NOT NULL,
              storage_path VARCHAR(1024) NOT NULL,
              size_bytes BIGINT DEFAULT 0,
              chunk_count INT DEFAULT 0,
              document_type VARCHAR(32) DEFAULT '',
              media_type VARCHAR(32) DEFAULT '',
              extracted_char_count INT DEFAULT 0,
              extraction_method VARCHAR(64) DEFAULT '',
              ocr_used BOOLEAN DEFAULT FALSE,
              ocr_status VARCHAR(32) DEFAULT '',
              multimodal_used BOOLEAN DEFAULT FALSE,
              multimodal_status VARCHAR(32) DEFAULT '',
              extraction_preview TEXT,
              status VARCHAR(32) DEFAULT 'ready',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_file_contents (
              document_id VARCHAR(64) PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              session_id VARCHAR(128) NULL,
              parsed_json JSON NULL,
              extraction_json JSON NULL,
              extracted_text MEDIUMTEXT NULL,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              INDEX idx_user_file_contents_user_session (user_id, session_id)
            )
            """
        )
        if not self.column_exists("user_file_contents", "session_id"):
            self.execute("ALTER TABLE user_file_contents ADD COLUMN session_id VARCHAR(128) NULL")
            self.execute("ALTER TABLE user_file_contents ADD INDEX idx_user_file_contents_user_session (user_id, session_id)")
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
              token_hash VARCHAR(128) PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              expires_at VARCHAR(64) NOT NULL
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
              user_id VARCHAR(64) PRIMARY KEY,
              settings_json JSON NOT NULL,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
              user_id VARCHAR(64) PRIMARY KEY,
              profile_json JSON NOT NULL,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_deletion_requests (
              user_id VARCHAR(64) PRIMARY KEY,
              status VARCHAR(32) NOT NULL,
              requested_at DATETIME NOT NULL,
              deletion_scheduled_at DATETIME NOT NULL,
              cancelled_at DATETIME NULL
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS user_activity (
              activity_id BIGINT AUTO_INCREMENT PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              action VARCHAR(64) NOT NULL,
              title VARCHAR(255) NOT NULL,
              detail TEXT,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              INDEX idx_user_activity_user_created (user_id, created_at)
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
              message_id BIGINT AUTO_INCREMENT PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              session_id VARCHAR(128) NOT NULL,
              role VARCHAR(32) NOT NULL,
              content MEDIUMTEXT NOT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              INDEX idx_chat_messages_session_created (user_id, session_id, created_at)
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS case_memories (
              user_id VARCHAR(64) NOT NULL,
              session_id VARCHAR(128) NOT NULL,
              memory_json JSON NOT NULL,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY(user_id, session_id)
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
              user_id VARCHAR(64) NOT NULL,
              session_id VARCHAR(128) NOT NULL,
              summary_text MEDIUMTEXT NOT NULL,
              message_count INT DEFAULT 0,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY(user_id, session_id)
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              conversation_id VARCHAR(128) NOT NULL,
              role VARCHAR(32) NOT NULL,
              content MEDIUMTEXT NOT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              INDEX idx_conversation_messages_user_conv_created (user_id, conversation_id, created_at),
              INDEX idx_conversation_messages_user_created (user_id, created_at)
            )
            """
        )
        if not self.column_exists("conversation_messages", "message_meta"):
            self.execute("ALTER TABLE conversation_messages ADD COLUMN message_meta JSON NULL")
        if not self.column_exists("conversation_messages", "solution_json"):
            self.execute("ALTER TABLE conversation_messages ADD COLUMN solution_json JSON NULL")
        if not self.column_exists("conversation_messages", "source_count"):
            self.execute("ALTER TABLE conversation_messages ADD COLUMN source_count INT DEFAULT 0")
        if not self.column_exists("conversation_messages", "question_text"):
            self.execute("ALTER TABLE conversation_messages ADD COLUMN question_text MEDIUMTEXT NULL")
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term_memory (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              scope VARCHAR(32) NOT NULL DEFAULT 'user',
              memory_type VARCHAR(32) NOT NULL,
              content TEXT NOT NULL,
              embedding JSON NULL,
              importance FLOAT DEFAULT 0.5,
              confidence FLOAT DEFAULT 0.5,
              status VARCHAR(32) NOT NULL DEFAULT 'active',
              source_conversation_id VARCHAR(128) NULL,
              source_message_id BIGINT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              last_used_at DATETIME NULL,
              INDEX idx_ltm_user_status_type (user_id, status, memory_type),
              INDEX idx_ltm_user_scope_status (user_id, scope, status),
              INDEX idx_ltm_user_updated (user_id, updated_at)
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_conflicts (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              user_id VARCHAR(64) NOT NULL,
              old_memory_id BIGINT NOT NULL,
              new_memory_json JSON NOT NULL,
              status VARCHAR(32) NOT NULL DEFAULT 'pending',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              resolved_at DATETIME NULL,
              INDEX idx_memory_conflicts_user_status (user_id, status, created_at)
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_jobs (
              user_id VARCHAR(64) PRIMARY KEY,
              last_message_id BIGINT DEFAULT 0,
              last_run_at DATETIME NULL,
              status VARCHAR(32) NOT NULL DEFAULT 'idle',
              error_message TEXT NULL,
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
        )

    def migrate_legacy_users_table(self) -> None:
        """把早期 users 表迁移到当前认证逻辑需要的列集合。"""
        columns = {row["Field"] for row in self.fetch_all("SHOW COLUMNS FROM users")}
        if "user_id" not in columns:
            self.execute("ALTER TABLE users ADD COLUMN user_id VARCHAR(64)")
            self.execute("UPDATE users SET user_id=REPLACE(UUID(), '-', '') WHERE user_id IS NULL OR user_id=''")
            self.execute("ALTER TABLE users ADD UNIQUE KEY idx_users_user_id (user_id)")
            columns.add("user_id")
        if "password_hash" not in columns:
            self.execute("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)")
            columns.add("password_hash")
            if "password" in columns:
                self.execute("UPDATE users SET password_hash=password WHERE (password_hash IS NULL OR password_hash='') AND password IS NOT NULL")
        if "email" not in columns:
            self.execute("ALTER TABLE users ADD COLUMN email VARCHAR(255)")
            self.execute("UPDATE users SET email=CONCAT(username, '@legacy.local') WHERE email IS NULL OR email=''")
            columns.add("email")
        if "created_at" not in columns:
            self.execute("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
            columns.add("created_at")
        if "updated_at" not in columns:
            self.execute("ALTER TABLE users ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
            columns.add("updated_at")
        if not self.index_exists("users", "idx_users_email"):
            self.execute("ALTER TABLE users ADD UNIQUE KEY idx_users_email (email)")

    def column_exists(self, table_name: str, column_name: str) -> bool:
        row = self.fetch_one(f"SHOW COLUMNS FROM `{table_name}` WHERE Field=%s", (column_name,))
        return bool(row)

    def index_exists(self, table_name: str, index_name: str) -> bool:
        row = self.fetch_one(f"SHOW INDEX FROM {table_name} WHERE Key_name=%s", (index_name,))
        return bool(row)

    def health(self) -> dict:
        row = self.fetch_one("SELECT 1 AS ok")
        return {"connected": bool(row), "backend": "mysql"}


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_expires_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class UserStore:
    """用户账号仓储，兼容新旧 users 表字段差异。"""

    def __init__(self, mysql: MySQLStore):
        self.mysql = mysql

    def create(self, username: str, email: str, password_hash: str) -> dict:
        user_id = uuid4().hex
        columns = {row["Field"] for row in self.mysql.fetch_all("SHOW COLUMNS FROM users")}
        fields = ["user_id", "username", "email", "password_hash"]
        values = [user_id, username, email, password_hash]
        placeholders = ["%s", "%s", "%s", "%s"]
        if "id" in columns:
            fields.insert(0, "id")
            values.insert(0, user_id)
            placeholders.insert(0, "%s")
        if "created_at" in columns:
            fields.append("created_at")
            placeholders.append("NOW(6)")
        if "updated_at" in columns:
            fields.append("updated_at")
            placeholders.append("NOW(6)")
        sql = f"INSERT INTO users({', '.join(fields)}) VALUES({', '.join(placeholders)})"
        self.mysql.execute(sql, tuple(values))
        return {"user_id": user_id, "username": username, "email": email}

    def by_username(self, username: str) -> dict | None:
        return self.mysql.fetch_one("SELECT * FROM users WHERE username=%s", (username,))

    def by_email(self, email: str) -> dict | None:
        return self.mysql.fetch_one("SELECT * FROM users WHERE email=%s", (email,))

    def by_id(self, user_id: str) -> dict | None:
        return self.mysql.fetch_one("SELECT * FROM users WHERE user_id=%s", (user_id,))


class AuthSessionStore:
    def __init__(self, mysql: MySQLStore):
        self.mysql = mysql

    def save(self, token: str, user_id: str, expires_at: str) -> None:
        self.mysql.execute(
            "REPLACE INTO user_sessions(token_hash, user_id, expires_at) VALUES(%s,%s,%s)",
            (token_hash(token), user_id, expires_at),
        )

    def get(self, token: str) -> dict | None:
        if not token:
            return None
        row = self.mysql.fetch_one("SELECT * FROM user_sessions WHERE token_hash=%s", (token_hash(token),))
        if not row:
            return None
        expires_at = parse_expires_at(row.get("expires_at", ""))
        if not expires_at or expires_at <= datetime.now(timezone.utc):
            self.delete(token)
            return None
        return row

    def delete(self, token: str) -> None:
        if token:
            self.mysql.execute("DELETE FROM user_sessions WHERE token_hash=%s", (token_hash(token),))


class FileStore:
    def __init__(self, mysql: MySQLStore):
        self.mysql = mysql

    def add(self, row: dict) -> dict:
        self.mysql.execute(
            """
            INSERT INTO user_files(document_id,user_id,session_id,file_name,storage_path,size_bytes,status)
            VALUES(%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                row["document_id"], row["user_id"], row.get("session_id"), row["file_name"],
                row["storage_path"], row.get("size_bytes", 0), row.get("status", "ready"),
            ),
        )
        return row

    def set_ready(self, document_id: str, chunk_count: int) -> None:
        self.mysql.execute("UPDATE user_files SET status='ready', chunk_count=%s WHERE document_id=%s", (chunk_count, document_id))

    def set_extraction_result(
        self,
        document_id: str,
        *,
        text: str,
        method: str,
        ocr_used: bool,
        multimodal_used: bool = False,
        multimodal_status: str = "not_used",
        document_type: str = "",
        media_type: str = "",
    ) -> None:
        preview = " ".join(text.strip().split())[:1000]
        self.mysql.execute(
            """
            UPDATE user_files
            SET extracted_char_count=%s, extraction_method=%s, ocr_used=%s, ocr_status=%s,
                multimodal_used=%s, multimodal_status=%s, document_type=%s, media_type=%s, extraction_preview=%s
            WHERE document_id=%s
            """,
            (
                len(text.strip()), method, ocr_used, "success" if ocr_used else "not_used",
                multimodal_used, multimodal_status, document_type, media_type, preview, document_id,
            ),
        )

    def save_parsed_content(
        self,
        document_id: str,
        user_id: str,
        session_id: str | None,
        extracted_text: str,
        parsed: dict,
        extraction: dict,
    ) -> None:
        """把一个文件的完整解析结果保存为 MySQL 中的一条材料记录。"""
        self.mysql.execute(
            """
            REPLACE INTO user_file_contents(document_id,user_id,session_id,parsed_json,extraction_json,extracted_text,updated_at)
            VALUES(%s,%s,%s,%s,%s,%s,NOW())
            """,
            (
                document_id,
                user_id,
                session_id,
                json.dumps(parsed or {}, ensure_ascii=False),
                json.dumps(extraction or {}, ensure_ascii=False),
                str(extracted_text or ""),
            ),
        )

    def get_parsed_content(self, user_id: str, document_id: str) -> dict | None:
        return self.mysql.fetch_one(
            "SELECT document_id,user_id,session_id,parsed_json,extraction_json,extracted_text FROM user_file_contents WHERE user_id=%s AND document_id=%s",
            (user_id, document_id),
        )

    def search_parsed_contents(self, user_id: str, document_ids: list[str], session_id: str | None, terms: list[str], limit: int = 20) -> list[dict]:
        """按用户和会话检索完整解析材料，不拆 chunk。"""
        if not document_ids:
            return []
        ids = [str(item) for item in document_ids if str(item).strip()]
        terms = [str(item).strip() for item in terms if str(item).strip()]
        if not ids:
            return []
        id_placeholders = ",".join(["%s"] * len(ids))
        params: list[Any] = [user_id, *ids]
        if session_id:
            session_clause = "AND (session_id=%s OR session_id IS NULL)"
            params.append(session_id)
        else:
            session_clause = ""
        term_clause = ""
        if terms:
            term_clause = "AND (" + " OR ".join(["extracted_text LIKE %s"] * len(terms)) + ")"
            params.extend([f"%{term}%" for term in terms])
        params.append(max(1, int(limit or 20)))
        return self.mysql.fetch_all(
            f"""
            SELECT document_id,user_id,session_id,extracted_text,parsed_json,extraction_json
            FROM user_file_contents
            WHERE user_id=%s AND document_id IN ({id_placeholders}) {session_clause}
              {term_clause}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            tuple(params),
        )

    def delete_parsed_content(self, document_id: str) -> None:
        self.mysql.execute("DELETE FROM user_file_contents WHERE document_id=%s", (document_id,))

    def set_failed(self, document_id: str, reason: str = "") -> None:
        self.mysql.execute(
            "UPDATE user_files SET status='failed' WHERE document_id=%s",
            (document_id,),
        )

    def set_stored(self, document_id: str, reason: str = "") -> None:
        self.mysql.execute(
            """
            UPDATE user_files
            SET status='stored', chunk_count=0,
                extraction_preview=CASE
                  WHEN COALESCE(extracted_char_count, 0) > 0 AND COALESCE(extraction_preview, '') <> '' THEN extraction_preview
                  ELSE %s
                END
            WHERE document_id=%s
            """,
            (str(reason or "").strip()[:1000], document_id),
        )

    def get(self, document_id: str) -> dict | None:
        return self.mysql.fetch_one("SELECT * FROM user_files WHERE document_id=%s", (document_id,))

    def list_for_user(self, user_id: str, session_id: str | None = None) -> list[dict]:
        if session_id:
            return self.mysql.fetch_all("SELECT * FROM user_files WHERE user_id=%s AND session_id=%s ORDER BY created_at DESC", (user_id, session_id))
        return self.mysql.fetch_all("SELECT * FROM user_files WHERE user_id=%s ORDER BY created_at DESC", (user_id,))

    def delete(self, document_id: str) -> None:
        self.delete_parsed_content(document_id)
        self.mysql.execute("DELETE FROM user_files WHERE document_id=%s", (document_id,))


DEFAULT_SETTINGS = {
    "include_web_default": True,
    "answer_detail": "standard",
    "show_retrieval_process": True,
    "auto_expand_professional": True,
    "enable_long_memory": False,
    "theme": "law_blue",
}

DEFAULT_PROFILE = {
    "occupation": "",
    "legal_level": "basic",
    "preferred_domains": [],
    "answer_style": "detailed",
    "citation_preference": True,
    "profile_summary": "",
}


class UserStateStore:
    def __init__(self, mysql: MySQLStore):
        self.mysql = mysql

    def normalize_settings(self, settings: dict | None = None) -> dict:
        source = {**DEFAULT_SETTINGS, **(settings or {})}
        detail = source.get("answer_detail")
        if detail not in {"concise", "standard", "detailed"}:
            detail = "standard"
        theme = source.get("theme")
        if theme not in {"law_blue", "government_green", "warm_gold", "indigo_ai", "cool_gray", "dark_night"}:
            theme = "law_blue"
        return {
            "include_web_default": bool(source.get("include_web_default")),
            "answer_detail": detail,
            "show_retrieval_process": bool(source.get("show_retrieval_process")),
            "auto_expand_professional": bool(source.get("auto_expand_professional")),
            "enable_long_memory": False,
            "cross_session_memory": bool(source.get("cross_session_memory")),
        }

    def get_setting(self, user_id: str, key: str):
        return self.get_settings(user_id).get(key)

    def set_setting(self, user_id: str, key: str, value) -> dict:
        settings = self.get_settings(user_id)
        settings[str(key or "")] = value
        return self.save_settings(user_id, settings)

    def get_settings(self, user_id: str) -> dict:
        row = self.mysql.fetch_one("SELECT settings_json FROM user_settings WHERE user_id=%s", (user_id,))
        if not row:
            return self.normalize_settings()
        value = row.get("settings_json") or "{}"
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {}
        return self.normalize_settings(value if isinstance(value, dict) else {})

    def save_settings(self, user_id: str, settings: dict) -> dict:
        normalized = self.normalize_settings(settings)
        self.mysql.execute(
            """
            REPLACE INTO user_settings(user_id, settings_json, updated_at)
            VALUES(%s, %s, NOW())
            """,
            (user_id, json.dumps(normalized, ensure_ascii=False)),
        )
        self.add_activity(user_id, "settings_updated", "偏好设置已更新", "用户更新了默认联网检索和回答展示设置")
        return normalized

    def normalize_profile(self, profile: dict | None = None) -> dict:
        source = {**DEFAULT_PROFILE, **(profile or {})}
        legal_level = source.get("legal_level")
        if legal_level not in {"basic", "intermediate", "professional"}:
            legal_level = "basic"
        answer_style = source.get("answer_style")
        if answer_style not in {"concise", "standard", "detailed"}:
            answer_style = "detailed"
        preferred_domains = []
        for item in source.get("preferred_domains") or []:
            value = str(item or "").strip()
            if value and value not in preferred_domains:
                preferred_domains.append(value)
        normalized = {
            "occupation": str(source.get("occupation") or "").strip(),
            "legal_level": legal_level,
            "preferred_domains": preferred_domains[:12],
            "answer_style": answer_style,
            "citation_preference": bool(source.get("citation_preference")),
            "profile_summary": str(source.get("profile_summary") or "").strip(),
        }
        if not normalized["profile_summary"]:
            occupation = normalized.get("occupation") or "普通用户"
            level_text = {
                "basic": "具有基础法律知识",
                "intermediate": "具有一定法律知识",
                "professional": "具有专业法律背景",
            }.get(normalized.get("legal_level"), "具有基础法律知识")
            domains = normalized.get("preferred_domains") or []
            domain_text = f"，关注{'、'.join(domains)}" if domains else ""
            style_text = {
                "concise": "偏好简明结论",
                "standard": "偏好清晰解释",
                "detailed": "偏好详细解释",
            }.get(normalized.get("answer_style"), "偏好详细解释")
            citation_text = "和具体法条" if normalized.get("citation_preference") else ""
            normalized["profile_summary"] = f"用户是{occupation}，{level_text}{domain_text}，{style_text}{citation_text}。"
        return normalized

    def get_profile(self, user_id: str) -> dict:
        row = self.mysql.fetch_one("SELECT profile_json FROM user_profiles WHERE user_id=%s", (user_id,))
        if not row:
            return self.normalize_profile()
        value = row.get("profile_json") or "{}"
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {}
        return self.normalize_profile(value if isinstance(value, dict) else {})

    def save_profile(self, user_id: str, profile: dict) -> dict:
        normalized = self.normalize_profile(profile)
        self.mysql.execute(
            """
            REPLACE INTO user_profiles(user_id, profile_json, updated_at)
            VALUES(%s, %s, NOW())
            """,
            (user_id, json.dumps(normalized, ensure_ascii=False)),
        )
        self.add_activity(user_id, "profile_updated", "用户画像已更新", "用户更新了职业、法律基础、关注领域和回答偏好")
        return normalized

    update_profile = save_profile

    @staticmethod
    def format_deletion(row: dict | None) -> dict:
        if not row or row.get("status") != "pending_deletion":
            return {"status": "active", "requested": False, "can_cancel": False}
        scheduled = row.get("deletion_scheduled_at")
        try:
            if isinstance(scheduled, datetime):
                scheduled_at = scheduled.replace(tzinfo=timezone.utc) if scheduled.tzinfo is None else scheduled.astimezone(timezone.utc)
            else:
                scheduled_at = datetime.fromisoformat(str(scheduled)).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return {"status": "active", "requested": False, "can_cancel": False}
        seconds_remaining = max(0, int((scheduled_at - datetime.now(timezone.utc)).total_seconds()))
        return {
            "status": "pending_deletion",
            "requested": True,
            "can_cancel": seconds_remaining > 0,
            "deletion_scheduled_at": scheduled_at.isoformat(),
            "seconds_remaining": seconds_remaining,
        }

    def get_deletion(self, user_id: str) -> dict:
        row = self.mysql.fetch_one("SELECT * FROM user_deletion_requests WHERE user_id=%s", (user_id,))
        return self.format_deletion(row)

    def request_deletion(self, user_id: str) -> dict:
        scheduled_at = datetime.now(timezone.utc) + timedelta(days=3)
        self.mysql.execute(
            """
            REPLACE INTO user_deletion_requests(user_id, status, requested_at, deletion_scheduled_at, cancelled_at)
            VALUES(%s, 'pending_deletion', NOW(), %s, NULL)
            """,
            (user_id, scheduled_at.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")),
        )
        self.add_activity(user_id, "deletion_requested", "账号注销申请已提交", "账号进入 3 天审核期，可在生效前取消")
        return self.get_deletion(user_id)

    def cancel_deletion(self, user_id: str) -> dict:
        self.mysql.execute(
            """
            UPDATE user_deletion_requests
            SET status='cancelled', cancelled_at=NOW()
            WHERE user_id=%s AND status='pending_deletion'
            """,
            (user_id,),
        )
        self.add_activity(user_id, "deletion_cancelled", "账号注销申请已取消", "账号恢复正常状态")
        return self.get_deletion(user_id)

    def add_activity(self, user_id: str, action: str, title: str, detail: str = "") -> None:
        self.mysql.execute(
            "INSERT INTO user_activity(user_id, action, title, detail) VALUES(%s,%s,%s,%s)",
            (user_id, action, title, detail),
        )

    def list_activity(self, user_id: str, limit: int = 20) -> list[dict]:
        safe_limit = min(max(int(limit or 20), 1), 100)
        return self.mysql.fetch_all(
            """
            SELECT action, title, detail, created_at
            FROM user_activity
            WHERE user_id=%s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, safe_limit),
        )


class ConversationHistoryStore:
    def __init__(self, mysql: MySQLStore):
        self.mysql = mysql

    def add(self, user_id: str, conversation_id: str, role: str, content: str, *, question_text: str = "", source_count: int = 0, message_meta: dict | None = None, solution_json: dict | None = None) -> int:
        with self.mysql.transaction() as conn:
            with conn.cursor() as cur:
                values = (user_id, conversation_id, role, content, question_text or None, int(source_count or 0), json.dumps(message_meta or {}, ensure_ascii=False) if message_meta else None, json.dumps(solution_json, ensure_ascii=False) if solution_json is not None else None)
                try:
                    cur.execute(
                        """
                        INSERT INTO conversation_messages(
                          user_id, conversation_id, role, content, question_text, source_count, message_meta, solution_json
                        )
                        VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        values,
                    )
                except (TypeError, ValueError):
                    cur.execute(
                        "INSERT INTO conversation_messages(user_id, conversation_id, role, content) VALUES(%s, %s, %s, %s)",
                        values[:4],
                    )
                return int(cur.lastrowid or 0)

    def list_recent(self, user_id: str, conversation_id: str, limit: int = 40) -> list[dict]:
        safe_limit = min(max(int(limit or 40), 1), 200)
        rows = self.mysql.fetch_all(
            """
            SELECT id, user_id, conversation_id, role, content, question_text, source_count, message_meta, solution_json, created_at
            FROM conversation_messages
            WHERE user_id=%s AND conversation_id=%s
            ORDER BY created_at ASC, id ASC
            LIMIT %s
            """,
            (user_id, conversation_id, safe_limit),
        )
        return [self.normalize_message(row) for row in rows]

    def list_all(self, user_id: str, conversation_id: str, limit: int | None = None) -> list[dict]:
        safe_limit = None if limit is None else min(max(int(limit or 0), 1), 2000)
        sql = """
            SELECT id, user_id, conversation_id, role, content, question_text, source_count, message_meta, solution_json, created_at
            FROM conversation_messages
            WHERE user_id=%s AND conversation_id=%s
            ORDER BY created_at ASC, id ASC
        """
        params: list[Any] = [user_id, conversation_id]
        if safe_limit is not None:
            sql += " LIMIT %s"
            params.append(safe_limit)
        rows = self.mysql.fetch_all(sql, tuple(params))
        return [dict(row) for row in rows]

    def list_conversations(self, user_id: str, limit: int = 100) -> list[dict]:
        safe_limit = min(max(int(limit or 100), 1), 500)
        rows = self.mysql.fetch_all(
            """
            SELECT conversation_id, COUNT(*) AS message_count, MAX(created_at) AS updated_at
            FROM conversation_messages
            WHERE user_id=%s
            GROUP BY conversation_id
            ORDER BY updated_at DESC, conversation_id DESC
            LIMIT %s
            """,
            (user_id, safe_limit),
        )
        return [
            {
                "conversation_id": row.get("conversation_id"),
                "message_count": int(row.get("message_count") or 0),
                "updated_at": row.get("updated_at"),
            }
            for row in rows
        ]

    def last_message_id(self, user_id: str, conversation_id: str) -> int:
        row = self.mysql.fetch_one(
            "SELECT MAX(id) AS last_id FROM conversation_messages WHERE user_id=%s AND conversation_id=%s",
            (user_id, conversation_id),
        )
        return int((row or {}).get("last_id") or 0)

    @staticmethod
    def normalize_message(row: dict) -> dict:
        normalized = dict(row)
        message_meta = normalized.get("message_meta")
        if isinstance(message_meta, str):
            try:
                message_meta = json.loads(message_meta)
            except json.JSONDecodeError:
                message_meta = {}
        solution_json = normalized.get("solution_json")
        if isinstance(solution_json, str):
            try:
                solution_json = json.loads(solution_json)
            except json.JSONDecodeError:
                solution_json = None
        answer = {
            "answer": str(normalized.get("content") or ""),
            "key_issues": [],
            "evidence_analysis": [],
            "defense_arguments": [],
            "action_steps": [],
            "document_checklist": [],
            "questions_to_confirm": [],
            "legal_basis": [],
            "related_cases": [],
            "suggestions": [],
            "risk_notice": "",
        }
        if isinstance(message_meta, dict):
            answer.update({k: message_meta.get(k, answer[k]) for k in answer.keys() if k != "answer"})
            if isinstance(message_meta.get("answer"), str):
                answer["answer"] = message_meta.get("answer") or answer["answer"]
        return {
            "id": normalized.get("id"),
            "role": normalized.get("role"),
            "content": str(normalized.get("content") or ""),
            "question": str(normalized.get("question_text") or ""),
            "answer": answer,
            "meta": message_meta if isinstance(message_meta, dict) else {},
            "sourceCount": int(normalized.get("source_count") or 0),
            "sourceFiles": [],
            "sources": [],
            "thinkingEnabled": False,
            "createdAt": normalized.get("created_at"),
            "solution": solution_json,
            "solutionCacheKey": str(solution_json.get("cache_key") if isinstance(solution_json, dict) else ""),
        }

    def latest_messages(self, user_id: str, limit: int = 20) -> list[dict]:
        safe_limit = min(max(int(limit or 20), 1), 100)
        rows = self.mysql.fetch_all(
            """
            SELECT conversation_id, role, content, question_text, source_count, message_meta, solution_json, created_at
            FROM conversation_messages
            WHERE user_id=%s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (user_id, safe_limit * 10),
        )
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            conversation_id = str(row.get("conversation_id") or "")
            if not conversation_id:
                continue
            grouped.setdefault(conversation_id, []).append(self.normalize_message({**row, "id": None}))
        summaries = []
        for conversation_id, messages in grouped.items():
            messages = list(reversed(messages))
            title = next((str(item.get("content") or "").strip() for item in messages if item.get("role") == "user" and str(item.get("content") or "").strip()), "新对话")
            summaries.append({
                "session_id": conversation_id,
                "title": title[:42],
                "message_count": len(messages),
                "updated_at": messages[-1].get("createdAt") if messages else None,
                "messages": messages[:80],
            })
        summaries.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return summaries[:safe_limit]

    def delete_session(self, user_id: str, conversation_id: str) -> None:
        session_id = str(conversation_id or "").strip()
        if not session_id:
            return
        self.mysql.execute("DELETE FROM conversation_summaries WHERE user_id=%s AND session_id=%s", (user_id, session_id))
        self.mysql.execute("DELETE FROM case_memories WHERE user_id=%s AND session_id=%s", (user_id, session_id))

    def count(self, user_id: str, conversation_id: str) -> int:
        row = self.mysql.fetch_one(
            "SELECT COUNT(*) AS count FROM conversation_messages WHERE user_id=%s AND conversation_id=%s",
            (user_id, conversation_id),
        )
        return int((row or {}).get("count") or 0)

    @classmethod
    def _compact_search_text(cls, value: object) -> str:
        return re.sub(r"[^\w一-鿿]+", "", str(value or "")).casefold()

    @classmethod
    def _search_terms(cls, query: str) -> list[str]:
        terms: list[str] = []
        for token in re.findall(r"[\w一-鿿]+", str(query or "")):
            compact = cls._compact_search_text(token)
            if len(compact) < 2:
                continue
            terms.append(compact)
            if re.search(r"[一-鿿]", compact) and len(compact) > 4:
                for size in (4, 3, 2):
                    for index in range(0, len(compact) - size + 1):
                        terms.append(compact[index:index + size])
        return list(dict.fromkeys(terms))[:24]

    def search_messages(self, user_id: str, conversation_id: str, query: str, limit: int = 8, scan_limit: int = 200) -> list[dict]:
        terms = self._search_terms(query)
        if not terms:
            return []
        safe_limit = min(max(int(limit or 8), 1), 20)
        safe_scan_limit = min(max(int(scan_limit or 200), safe_limit), 1000)
        rows = self.mysql.fetch_all(
            """
            SELECT id, user_id, conversation_id, role, content, created_at
            FROM conversation_messages
            WHERE user_id=%s AND conversation_id=%s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (user_id, conversation_id, safe_scan_limit),
        )
        query_norm = self._compact_search_text(query)
        scored: list[dict] = []
        for row_rank, row in enumerate(rows):
            content = str(row.get("content") or "")
            haystack = self._compact_search_text(content)
            if not haystack:
                continue
            matched = [term for term in terms if term in haystack]
            if query_norm and len(query_norm) >= 4 and query_norm in haystack:
                score = 0.98
                reason = "original_exact"
            elif matched:
                longest = max(len(term) for term in matched)
                score = 0.72 + min(0.18, len(matched) * 0.035) + min(0.06, longest * 0.008)
                reason = "original_keyword"
            else:
                continue
            scored.append({
                **row,
                "score": round(min(0.96, score), 4),
                "retrieval_reason": reason,
                "matched_terms": matched[:8],
                "_row_rank": row_rank,
            })
        scored.sort(key=lambda row: (float(row.get("score", 0) or 0), -int(row.get("_row_rank", 0))), reverse=True)
        return [{key: value for key, value in row.items() if key != "_row_rank"} for row in scored[:safe_limit]]

class LongTermMemoryStore:
    def __init__(self, mysql: MySQLStore):
        self.mysql = mysql

    @staticmethod
    def _loads(value: Any, fallback: Any) -> Any:
        if value in (None, ""):
            return fallback
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except json.JSONDecodeError:
            return fallback

    def normalize_row(self, row: dict) -> dict:
        normalized = dict(row)
        normalized["embedding"] = self._loads(normalized.get("embedding"), None)
        normalized["importance"] = float(normalized.get("importance") or 0)
        normalized["confidence"] = float(normalized.get("confidence") or 0)
        return normalized

    def add(self, row: dict) -> int:
        with self.mysql.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO long_term_memory(
                      user_id, scope, memory_type, content, embedding, importance, confidence, status,
                      source_conversation_id, source_message_id
                    )
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        row["user_id"], row.get("scope", "user"), row["memory_type"], row["content"],
                        json.dumps(row.get("embedding"), ensure_ascii=False) if row.get("embedding") is not None else None,
                        float(row.get("importance", 0.5)), float(row.get("confidence", 0.5)), row.get("status", "active"),
                        row.get("source_conversation_id"), row.get("source_message_id"),
                    ),
                )
                return int(cur.lastrowid or 0)

    def get(self, user_id: str, memory_id: int) -> dict | None:
        row = self.mysql.fetch_one(
            "SELECT * FROM long_term_memory WHERE user_id=%s AND id=%s",
            (user_id, int(memory_id)),
        )
        return self.normalize_row(row) if row else None

    def list_active(self, user_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        safe_limit = min(max(int(limit or 100), 1), 500)
        safe_offset = max(int(offset or 0), 0)
        rows = self.mysql.fetch_all(
            """
            SELECT * FROM long_term_memory
            WHERE user_id=%s AND status='active'
            ORDER BY importance DESC, updated_at DESC
            LIMIT %s OFFSET %s
            """,
            (user_id, safe_limit, safe_offset),
        )
        return [self.normalize_row(row) for row in rows]

    def list_for_retrieval(
        self,
        user_id: str,
        scope: str = "user",
        limit: int = 200,
        conversation_id: str | None = None,
        memory_types: set[str] | tuple[str, ...] | list[str] | None = None,
    ) -> list[dict]:
        safe_limit = min(max(int(limit or 200), 1), 1000)
        sql = """
            SELECT * FROM long_term_memory
            WHERE user_id=%s AND scope=%s AND status='active' AND embedding IS NOT NULL
        """
        params: list[Any] = [user_id, scope]
        allowed_types = [str(item) for item in (memory_types or []) if str(item or "").strip()]
        if allowed_types:
            placeholders = ",".join(["%s"] * len(allowed_types))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(allowed_types)
        if conversation_id:
            sql += " AND source_conversation_id=%s"
            params.append(str(conversation_id))
        sql += " ORDER BY importance DESC, COALESCE(last_used_at, updated_at) DESC LIMIT %s"
        params.append(safe_limit)
        rows = self.mysql.fetch_all(sql, tuple(params))
        return [self.normalize_row(row) for row in rows]

    def list_by_type(self, user_id: str, memory_type: str, limit: int = 100) -> list[dict]:
        safe_limit = min(max(int(limit or 100), 1), 500)
        rows = self.mysql.fetch_all(
            """
            SELECT * FROM long_term_memory
            WHERE user_id=%s AND memory_type=%s AND status='active'
            ORDER BY importance DESC, updated_at DESC
            LIMIT %s
            """,
            (user_id, memory_type, safe_limit),
        )
        return [self.normalize_row(row) for row in rows]

    def mark_used(self, user_id: str, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join(["%s"] * len(memory_ids))
        self.mysql.execute(
            f"UPDATE long_term_memory SET last_used_at=NOW() WHERE user_id=%s AND id IN ({placeholders})",
            tuple([user_id, *[int(item) for item in memory_ids]]),
        )

    def latest_source_message_id(self, user_id: str, conversation_id: str) -> int:
        row = self.mysql.fetch_one(
            """
            SELECT MAX(source_message_id) AS last_id
            FROM long_term_memory
            WHERE user_id=%s AND source_conversation_id=%s AND status='active'
            """,
            (user_id, conversation_id),
        )
        return int((row or {}).get("last_id") or 0)

    def delete(self, user_id: str, memory_id: int) -> bool:
        affected = self.mysql.execute(
            "UPDATE long_term_memory SET status='deprecated' WHERE user_id=%s AND id=%s AND status='active'",
            (user_id, int(memory_id)),
        )
        return bool(affected)


class MemoryConflictStore:
    def __init__(self, mysql: MySQLStore):
        self.mysql = mysql

    def add(self, user_id: str, old_memory_id: int, new_memory: dict) -> int:
        with self.mysql.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_conflicts(user_id, old_memory_id, new_memory_json, status)
                    VALUES(%s, %s, %s, 'pending')
                    """,
                    (user_id, int(old_memory_id), json.dumps(new_memory, ensure_ascii=False)),
                )
                return int(cur.lastrowid or 0)

    def list_pending(self, user_id: str, limit: int = 50) -> list[dict]:
        safe_limit = min(max(int(limit or 50), 1), 200)
        rows = self.mysql.fetch_all(
            """
            SELECT c.*, m.content AS old_memory_content, m.memory_type AS old_memory_type
            FROM memory_conflicts c
            LEFT JOIN long_term_memory m ON m.id=c.old_memory_id AND m.user_id=c.user_id
            WHERE c.user_id=%s AND c.status='pending'
            ORDER BY c.created_at DESC
            LIMIT %s
            """,
            (user_id, safe_limit),
        )
        normalized = []
        for row in rows:
            item = dict(row)
            item["new_memory"] = LongTermMemoryStore._loads(item.pop("new_memory_json", None), {})
            normalized.append(item)
        return normalized

    def get_pending(self, user_id: str, conflict_id: int) -> dict | None:
        rows = [row for row in self.list_pending(user_id, limit=200) if int(row.get("id") or 0) == int(conflict_id)]
        return rows[0] if rows else None

    def resolve(self, user_id: str, conflict_id: int, status: str) -> None:
        self.mysql.execute(
            "UPDATE memory_conflicts SET status=%s, resolved_at=NOW() WHERE user_id=%s AND id=%s AND status='pending'",
            (status, user_id, int(conflict_id)),
        )
UserRepository = UserStore
SessionRepository = AuthSessionStore
FileRepository = FileStore
UserStateRepository = UserStateStore
