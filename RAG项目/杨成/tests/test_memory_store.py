import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from memory_store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_recent_messages_are_isolated_and_limited_to_rounds(self):
        for index in range(8):
            self.store.add_message("session-a", "user", f"问题{index}")
            self.store.add_message("session-a", "assistant", f"回答{index}")
        self.store.add_message("session-b", "user", "另一个会话")

        messages = self.store.get_recent_messages("session-a", max_rounds=6)

        self.assertEqual(len(messages), 12)
        self.assertEqual(messages[0], {"role": "user", "content": "问题2"})
        self.assertEqual(messages[-1], {"role": "assistant", "content": "回答7"})
        self.assertEqual(self.store.get_recent_messages("session-b"), [{"role": "user", "content": "另一个会话"}])

    def test_long_term_memory_crud_is_scoped_to_user(self):
        memory = self.store.create_memory("user-a", "对青霉素过敏", category="过敏史")
        self.store.create_memory("user-b", "偏好简洁回答", category="偏好")

        memories = self.store.list_memories("user-a")

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content"], "对青霉素过敏")
        self.assertEqual(memories[0]["category"], "过敏史")
        self.assertTrue(self.store.delete_memory("user-a", memory["id"]))
        self.assertFalse(self.store.delete_memory("user-b", memory["id"]))
        self.assertEqual(self.store.list_memories("user-a"), [])

    def test_sessions_are_scoped_and_messages_keep_order(self):
        self.store.ensure_session("user-a", "session-a", "第一次咨询")
        self.store.ensure_session("user-b", "session-b", "其他用户")
        self.store.add_message("session-a", "user", "问题一")
        self.store.add_message("session-a", "assistant", "回答一")

        sessions = self.store.list_sessions("user-a")
        messages = self.store.get_messages("session-a")

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "session-a")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["content"], "回答一")
        self.assertEqual(self.store.get_session("session-a", "user-b"), None)
        self.assertEqual(self.store.get_messages("session-b", "user-a"), None)

        with self.assertRaises(ValueError):
            self.store.add_message("session-a", "user", "")
        with self.assertRaises(ValueError):
            self.store.create_memory("user-a", "", category="其他")


if __name__ == "__main__":
    unittest.main()
