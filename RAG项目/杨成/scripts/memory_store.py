import sqlite3
from pathlib import Path


class MemoryStore:
    def __init__(self, database_path):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions(user_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '其他',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user_id_id
                ON memories(user_id, id);
            """
        )
        self.connection.commit()

    @staticmethod
    def _require_text(value, field_name):
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        return text

    def ensure_session(self, user_id, session_id, title="新会话"):
        user_id = self._require_text(user_id, "user_id")
        session_id = self._require_text(session_id, "session_id")
        title = self._require_text(title, "title")[:80]
        self.connection.execute(
            """
            INSERT INTO sessions (session_id, user_id, title)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, user_id, title),
        )
        self.connection.commit()
        return self.get_session(session_id, user_id)

    def get_session(self, session_id, user_id):
        session_id = self._require_text(session_id, "session_id")
        user_id = self._require_text(user_id, "user_id")
        row = self.connection.execute(
            """
            SELECT session_id, user_id, title, created_at, updated_at
            FROM sessions WHERE session_id = ? AND user_id = ?
            """,
            (session_id, user_id),
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, user_id):
        user_id = self._require_text(user_id, "user_id")
        rows = self.connection.execute(
            """
            SELECT session_id, user_id, title, created_at, updated_at
            FROM sessions WHERE user_id = ? ORDER BY updated_at DESC, rowid DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, session_id, role, content):
        session_id = self._require_text(session_id, "session_id")
        role = self._require_text(role, "role")
        content = self._require_text(content, "content")
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        cursor = self.connection.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        self.connection.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,),
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_messages(self, session_id, user_id=None):
        session_id = self._require_text(session_id, "session_id")
        if user_id is not None and self.get_session(session_id, user_id) is None:
            return None
        rows = self.connection.execute(
            """
            SELECT id, role, content, created_at FROM messages
            WHERE session_id = ? ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_messages(self, session_id, max_rounds=6):
        session_id = self._require_text(session_id, "session_id")
        if max_rounds < 1:
            return []
        rows = self.connection.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, max_rounds * 2),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def create_memory(self, user_id, content, category="其他"):
        user_id = self._require_text(user_id, "user_id")
        content = self._require_text(content, "content")
        category = self._require_text(category, "category")
        cursor = self.connection.execute(
            "INSERT INTO memories (user_id, content, category) VALUES (?, ?, ?)",
            (user_id, content, category),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT id, user_id, content, category, created_at FROM memories WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        return dict(row)

    def list_memories(self, user_id):
        user_id = self._require_text(user_id, "user_id")
        rows = self.connection.execute(
            """
            SELECT id, user_id, content, category, created_at
            FROM memories WHERE user_id = ? ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_memory(self, user_id, memory_id):
        user_id = self._require_text(user_id, "user_id")
        cursor = self.connection.execute(
            "DELETE FROM memories WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def close(self):
        self.connection.close()
