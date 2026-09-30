"""Data-layer behavior without requiring running infrastructure."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fakeredis

from app import db
from app.memory import ConversationMemory


class MySqlCompatibleRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"RAG_TEST_MODE": "1"})
        self.path = patch.object(db, "DB_PATH", Path(self.temp.name) / "test.db")
        self.env.start()
        self.path.start()
        db.init_db()

    def tearDown(self):
        self.path.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_passwords_are_salted_and_login_is_parameterized(self):
        first = db.hash_password("same-password")
        second = db.hash_password("same-password")
        self.assertNotEqual(first, second)
        self.assertTrue(db.password_matches("same-password", first))
        user_id = db.create_user("alice' OR 1=1 --", "safe-password")
        self.assertIsInstance(user_id, int)
        self.assertIsNone(db.verify_user("alice", "safe-password"))
        self.assertEqual(db.verify_user("alice' OR 1=1 --", "safe-password")["id"], user_id)
        self.assertIsNone(db.create_user("alice' OR 1=1 --", "different"))

    def test_legacy_sha256_is_upgraded_after_login(self):
        legacy = hashlib.sha256(b"old-password").hexdigest()
        with db.transaction() as conn:
            db.execute(conn, "INSERT INTO users(username, password_hash) VALUES (?, ?)",
                       ("legacy", legacy)).close()
        self.assertIsNotNone(db.verify_user("legacy", "old-password"))
        with db.transaction() as conn:
            cursor = db.execute(conn, "SELECT password_hash FROM users WHERE username = ?", ("legacy",))
            upgraded = cursor.fetchone()["password_hash"]
            cursor.close()
        self.assertTrue(upgraded.startswith("pbkdf2_sha256$"))


class RedisMemoryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RAG_TEST_MODE": "1", "RAG_MEMORY_TTL_SECONDS": "120"})
        self.env.start()
        self.memory = ConversationMemory(max_turns=2)

    def tearDown(self):
        self.env.stop()

    def test_turns_are_isolated_trimmed_expiring_and_clearable(self):
        self.memory.add_turn(1, 1, "q1", "a1")
        self.memory.add_turn(2, 1, "other", "reply")
        self.memory.add_turn(1, 1, "q2", "a2")
        self.memory.add_turn(1, 1, "q3", "a3")
        history = self.memory.history(1, 1)
        self.assertNotIn("q1", history)
        self.assertIn("q2", history)
        self.assertIn("q3", history)
        self.assertNotIn("other", history)
        self.assertGreater(self.memory.client.ttl(self.memory.key(1, 1)), 0)
        self.memory.clear(1, 1)
        self.assertIn("无历史对话", self.memory.history(1, 1))

    def test_fake_backend_requires_explicit_test_mode(self):
        self.assertEqual(self.memory.backend, "fakeredis")
        with patch.dict(os.environ, {"RAG_TEST_MODE": "0", "RAG_REDIS_URL": "redis://127.0.0.1:1/0"}):
            with self.assertRaises(Exception):
                ConversationMemory()


if __name__ == "__main__":
    unittest.main()
