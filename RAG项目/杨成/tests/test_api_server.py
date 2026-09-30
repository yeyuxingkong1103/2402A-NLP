import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi.testclient import TestClient

import api_server


class ApiServerTests(unittest.TestCase):
    def setUp(self):
        self.database = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.database.close()
        api_server.configure_memory_store(self.database.name)

    def tearDown(self):
        api_server.close_memory_store()
        Path(self.database.name).unlink(missing_ok=True)

    def test_health_returns_ok(self):
        client = TestClient(api_server.app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_home_returns_frontend_html(self):
        client = TestClient(api_server.app)

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("基层高血压 RAG 专家系统", response.text)

    def test_home_returns_session_sidebar_without_memory_ui(self):
        client = TestClient(api_server.app)

        response = client.get("/")

        self.assertIn("new-session-button", response.text)
        self.assertIn("session-list", response.text)
        self.assertIn("/sessions", response.text)
        self.assertIn("answer_mode", response.text)
        self.assertNotIn("memory-form", response.text)
        self.assertNotIn("/memories", response.text)
        self.assertNotIn("左侧保留短期会话历史", response.text)
        self.assertNotIn("长期记忆仍在后端生效", response.text)

    def test_home_uses_medical_professional_visual_layout(self):
        client = TestClient(api_server.app)

        response = client.get("/")

        self.assertIn('class="brand"', response.text)
        self.assertIn('class="brand-mark"', response.text)
        self.assertIn('class="system-card"', response.text)
        self.assertIn('class="content-header"', response.text)
        self.assertIn('class="mode-pill"', response.text)
        self.assertIn("source-index", response.text)
        self.assertIn("--accent: #0c7c84", response.text)

    def test_home_uses_compact_question_composer(self):
        client = TestClient(api_server.app)

        response = client.get("/")

        self.assertIn("question-composer", response.text)
        self.assertIn("justify-self: end", response.text)
        self.assertIn("width: fit-content", response.text)
        self.assertIn("display: inline-block", response.text)
        self.assertIn("padding: 8px 12px", response.text)
        self.assertIn("line-height: 1.5", response.text)
        self.assertIn("justify-items: start", response.text)
        self.assertIn("align-items: start", response.text)
        self.assertIn("question.value = '';", response.text)

    def test_design_preview_contains_three_switchable_visual_options(self):
        client = TestClient(api_server.app)

        response = client.get("/design-preview")

        self.assertEqual(response.status_code, 200)
        self.assertIn("方案 A · 医疗专业版", response.text)
        self.assertIn("方案 B · 现代 AI 对话版", response.text)
        self.assertIn("方案 C · 深色专家工作台", response.text)
        self.assertIn("data-theme=\"medical\"", response.text)
        self.assertIn("data-theme=\"modern\"", response.text)
        self.assertIn("data-theme=\"workbench\"", response.text)
        self.assertIn("theme-picker", response.text)

    def test_chat_returns_answer_sources_latency_and_mode(self):
        fake_result = {
            "answer": "氨氯地平禁忌证为快速性心律失常。[来源]",
            "retrieved": [
                {
                    "section_path": "5 高血压治疗",
                    "chunk_type": "table_row",
                    "rerank_score": 0.98,
                    "text_preview": "ignored",
                }
            ],
            "latency": {"total_ms": 1234},
            "answer_mode": "rag",
            "notice": "本回答基于本地知识库检索结果生成。",
        }

        with patch.object(api_server, "get_rag_runtime", return_value="runtime"):
            with patch.object(api_server, "answer_with_runtime", return_value=fake_result):
                client = TestClient(api_server.app)
                response = client.post("/chat", json={"question": "氨氯地平的禁忌证是什么？"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "answer": "氨氯地平禁忌证为快速性心律失常。[来源]",
                "sources": [
                    {
                        "section_path": "5 高血压治疗",
                        "chunk_type": "table_row",
                        "rerank_score": 0.98,
                    }
                ],
                "latency_ms": 1234,
                "answer_mode": "rag",
                "notice": "本回答基于本地知识库检索结果生成。",
            },
        )

    def test_sessions_and_messages_are_listed_for_owner(self):
        with patch.object(api_server, "get_rag_runtime", return_value="runtime"):
            with patch.object(api_server, "answer_with_runtime") as fake_answer:
                fake_answer.return_value = {
                    "answer": "回答 [章节]",
                    "retrieved": [],
                    "latency": {"total_ms": 10},
                    "answer_mode": "rag",
                    "notice": "本回答基于本地知识库检索结果生成。",
                }
                client = TestClient(api_server.app)
                chat = client.post(
                    "/chat",
                    json={"question": "什么是高血压？", "session_id": "session-a", "user_id": "user-a"},
                )
                sessions = client.get("/sessions", params={"user_id": "user-a"})
                messages = client.get("/sessions/session-a/messages", params={"user_id": "user-a"})
                forbidden = client.get("/sessions/session-a/messages", params={"user_id": "user-b"})

        self.assertEqual(chat.status_code, 200)
        self.assertEqual(sessions.status_code, 200)
        self.assertEqual(sessions.json()[0]["session_id"], "session-a")
        self.assertEqual(messages.status_code, 200)
        self.assertEqual(messages.json()[0]["content"], "什么是高血压？")
        self.assertEqual(messages.json()[1]["content"], "回答 [章节]")
        self.assertEqual(forbidden.status_code, 404)

    def test_memories_can_be_created_listed_and_deleted(self):
        client = TestClient(api_server.app)

        response = client.post(
            "/memories",
            json={"user_id": "user-a", "content": "对青霉素过敏", "category": "过敏史"},
        )

        self.assertEqual(response.status_code, 201)
        memory_id = response.json()["id"]
        listed = client.get("/memories", params={"user_id": "user-a"})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0]["id"], memory_id)
        deleted = client.delete(f"/memories/{memory_id}", params={"user_id": "user-a"})
        self.assertEqual(deleted.status_code, 204)

    def test_chat_reuses_short_term_context_for_same_session(self):
        calls = []

        def fake_answer(runtime, question, conversation_context="", long_term_memory=""):
            calls.append((question, conversation_context, long_term_memory))
            return {
                "answer": f"回答：{question} [章节]",
                "retrieved": [],
                "latency": {"total_ms": 10},
                "answer_mode": "rag",
                "notice": "本回答基于本地知识库检索结果生成。",
            }

        with patch.object(api_server, "get_rag_runtime", return_value="runtime"):
            with patch.object(api_server, "answer_with_runtime", side_effect=fake_answer):
                client = TestClient(api_server.app)
                first = client.post(
                    "/chat",
                    json={"question": "什么是高血压？", "session_id": "session-a", "user_id": "user-a"},
                )
                second = client.post(
                    "/chat",
                    json={"question": "那诊断标准呢？", "session_id": "session-a", "user_id": "user-a"},
                )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("什么是高血压？", calls[1][1])
        self.assertIn("回答：什么是高血压？", calls[1][1])


if __name__ == "__main__":
    unittest.main()
