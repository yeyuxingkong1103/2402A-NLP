"""Mock tests for app/single_app.py; no real service or model is contacted."""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import fakeredis
import httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError, APITimeoutError, OpenAIError


PROJECT_ROOT = Path(
    os.environ.get("RAG_TEST_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
SOURCE = PROJECT_ROOT / "app" / "single_app.py"


class StubMilvusClient:
    def __init__(self, *args, **kwargs):
        pass


class StubSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass


class StubCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass


_MISSING = object()
_module_names = ("pymilvus", "sentence_transformers")
_saved_modules = {name: sys.modules.get(name, _MISSING) for name in _module_names}
_env_names = ("RAG_ENV_FILE", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
_saved_env = {name: os.environ.get(name, _MISSING) for name in _env_names}
with tempfile.TemporaryDirectory() as _tempdir:
    _empty_env = Path(_tempdir) / "empty.env"
    _empty_env.write_text("", encoding="utf-8")
    os.environ.update({
        "RAG_ENV_FILE": str(_empty_env),
        "LLM_API_KEY": "mock-key",
        "LLM_BASE_URL": "https://mock.invalid/v1",
        "LLM_MODEL": "mock-model",
    })
    sys.modules["pymilvus"] = types.SimpleNamespace(MilvusClient=StubMilvusClient)
    sys.modules["sentence_transformers"] = types.SimpleNamespace(
        CrossEncoder=StubCrossEncoder,
        SentenceTransformer=StubSentenceTransformer,
    )
    try:
        spec = importlib.util.spec_from_file_location("single_app_under_test", SOURCE)
        single = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(single)
    finally:
        for name, value in _saved_modules.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        for name, value in _saved_env.items():
            if value is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class FakeEmbedding:
    def encode(self, texts, **kwargs):
        return [FakeArray([0.1, 0.2])]


class FakeArray(list):
    def tolist(self):
        return list(self)


class FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, _tokens):
        return self.scores


class FakeMilvus:
    def __init__(self, hits):
        self.hits = hits

    def search(self, *args, **kwargs):
        return [self.hits]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def hit(doc_id, distance, text=None):
    return {
        "id": doc_id,
        "distance": distance,
        "entity": {
            "id": doc_id,
            "text": text or f"片段{doc_id}",
            "source": "guide.pdf",
            "chunk_index": doc_id,
        },
    }


def bare_rag(vector_hits, bm25_scores, reranker=None):
    rag = object.__new__(single.RAG)
    rag.embedder = FakeEmbedding()
    rag.milvus = FakeMilvus(vector_hits)
    rag.rows = [
        {"id": index + 1, "text": f"片段{index + 1}", "source": "guide.pdf", "chunk_index": index + 1}
        for index in range(len(bm25_scores))
    ]
    rag.bm25 = FakeBM25(bm25_scores)
    rag.reranker = reranker
    return rag


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.scope = patch.dict(os.environ, {"RAG_KNOWLEDGE_SCOPE_TERMS": "高血压"})
        self.scope.start()

    def tearDown(self):
        self.scope.stop()

    def test_rrf_fuses_duplicate_and_respects_top_k(self):
        rag = bare_rag([hit(1, 0.91), hit(2, 0.82)], [0.0, 8.0, 7.0])
        results = rag.retrieve("高血压如何管理", top_k=2, use_rerank=False)
        self.assertEqual([2, 1], [item["entity"]["id"] for item in results])
        self.assertGreater(results[0]["rrf_score"], results[1]["rrf_score"])
        self.assertEqual(8.0, results[0]["bm25_score"])
        self.assertLessEqual(len(results), 2)

    def test_reranker_changes_order_and_can_be_disabled(self):
        reranker = FakeReranker([0.1, 0.9])
        rag = bare_rag([hit(1, 0.91), hit(2, 0.82)], [0.0, 0.0], reranker)
        ranked = rag.retrieve("高血压", top_k=2)
        self.assertEqual([2, 1], [item["entity"]["id"] for item in ranked])
        self.assertEqual(1, len(reranker.calls))

        reranker.calls.clear()
        unranked = rag.retrieve("高血压", top_k=2, use_rerank=False)
        self.assertEqual([1, 2], [item["entity"]["id"] for item in unranked])
        self.assertEqual([], reranker.calls)

    def test_scope_and_low_score_return_no_evidence(self):
        rag = bare_rag([hit(1, 0.9)], [5.0])
        self.assertEqual([], rag.retrieve("感冒怎么办"))
        low = bare_rag([hit(1, 0.1)], [0.0])
        self.assertEqual([], low.retrieve("高血压怎么办"))

    def test_top_k_rejects_invalid_values(self):
        rag = bare_rag([hit(1, 0.9), hit(2, 0.8)], [0.0, 0.0])
        for invalid in (0, -1, 11, 1.5, True, "4"):
            with self.subTest(top_k=invalid), self.assertRaisesRegex(ValueError, "1到10"):
                rag.retrieve("高血压", top_k=invalid, use_rerank=False)

    def test_top_k_accepts_boundaries(self):
        rag = bare_rag([hit(1, 0.9), hit(2, 0.8)], [0.0, 0.0])
        self.assertEqual(1, len(rag.retrieve("高血压", top_k=1, use_rerank=False)))
        self.assertEqual(2, len(rag.retrieve("高血压", top_k=10, use_rerank=False)))

    def test_constructor_honors_disabled_reranker(self):
        milvus = Mock()
        milvus.query.return_value = [
            {"id": 1, "text": "高血压资料", "source": "guide.pdf", "chunk_index": 1}
        ]
        with patch.object(single, "OpenAI", return_value=Mock()), patch.object(
            single, "SentenceTransformer", return_value=FakeEmbedding()
        ), patch.object(single, "MilvusClient", return_value=milvus), patch.object(
            single, "CrossEncoder"
        ) as cross_encoder, patch.dict(os.environ, {
            "RAG_RERANK_ENABLED": "false",
            "LLM_API_KEY": "mock-key",
            "LLM_BASE_URL": "https://mock.invalid/v1",
            "LLM_MODEL": "mock-model",
        }):
            rag = single.RAG(with_memory=False)
        self.assertIsNone(rag.reranker)
        self.assertEqual("disabled", rag.rerank_state)
        cross_encoder.assert_not_called()


class GenerationTests(unittest.TestCase):
    def response(self, text):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        )

    def test_no_sources_removes_fake_references_and_passes_options(self):
        rag = object.__new__(single.RAG)
        create = Mock(return_value=self.response("**建议** [资料1] 观察。"))
        rag.llm = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        rag.model = "deepseek-mock"
        rag.options = {"extra_body": {"thinking": {"type": "disabled"}}}

        answer = rag.generate("感冒怎么办", "（无历史对话）", [])
        self.assertTrue(answer.startswith("当前知识库未覆盖这个主题"))
        self.assertNotIn("[资料1]", answer)
        self.assertNotIn("*", answer)
        kwargs = create.call_args.kwargs
        self.assertEqual("deepseek-mock", kwargs["model"])
        self.assertEqual(0.3, kwargs["temperature"])
        self.assertEqual(768, kwargs["max_tokens"])
        self.assertEqual({"thinking": {"type": "disabled"}}, kwargs["extra_body"])

    def test_empty_model_answer_raises_openai_error(self):
        rag = object.__new__(single.RAG)
        rag.llm = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=Mock(return_value=self.response("   ")))
            )
        )
        rag.model = "mock"
        rag.options = {}
        with self.assertRaises(OpenAIError):
            rag.generate("问题", "历史", [])


class MemoryTests(unittest.TestCase):
    def test_memory_isolated_trimmed_and_expiring(self):
        server = fakeredis.FakeServer()

        def client_factory(*args, **kwargs):
            return fakeredis.FakeRedis(server=server, decode_responses=True)

        with patch.object(single.redis.Redis, "from_url", side_effect=client_factory), patch.dict(
            os.environ, {"RAG_MEMORY_TTL_SECONDS": "120"}
        ):
            memory = single.Memory()
            for index in range(12):
                memory.add(1, 1, f"问题{index}", f"回答{index}")
            memory.add(2, 1, "另一个用户", "独立回答")

            messages = memory.client.lrange("chat:1:1", 0, -1)
            self.assertEqual(20, len(messages))
            self.assertNotIn("问题0", "\n".join(messages))
            self.assertIn("问题11", "\n".join(messages))
            self.assertEqual(["用户：另一个用户", "助手：独立回答"], memory.client.lrange("chat:2:1", 0, -1))
            ttl = memory.client.ttl("chat:1:1")
            self.assertGreater(ttl, 0)
            self.assertLessEqual(ttl, 120)
            memory.clear(1, 1)
            self.assertEqual("（无历史对话）", memory.history(1, 1))


class FailingCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, sql, values):
        raise RuntimeError("database write failed")


class TrackingConnection:
    def __init__(self):
        self.rolled_back = False
        self.closed = False

    def cursor(self, dictionary=False):
        return FailingCursor()

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class DatabaseTests(unittest.TestCase):
    def test_query_rolls_back_and_closes_after_failure(self):
        connection = TrackingConnection()
        with patch.object(single, "database", return_value=connection), self.assertRaises(RuntimeError):
            single.query("INSERT INTO users(username) VALUES(%s)", ("alice",))
        self.assertTrue(connection.rolled_back)
        self.assertTrue(connection.closed)


class SQLiteCursorAdapter:
    def __init__(self, cursor):
        self.cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.cursor.close()

    def execute(self, sql, values=()):
        translated = sql.replace("INSERT IGNORE", "INSERT OR IGNORE").replace("%s", "?")
        self.cursor.execute(translated, values)

    def fetchone(self):
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def fetchall(self):
        return [dict(row) for row in self.cursor.fetchall()]

    @property
    def lastrowid(self):
        return self.cursor.lastrowid


class SQLiteConnectionAdapter:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row

    def cursor(self, dictionary=False):
        return SQLiteCursorAdapter(self.connection.cursor())

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        self.connection.close()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "accounts.sqlite"
        raw = sqlite3.connect(self.db_path)
        raw.executescript(
            """
            CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT);
            CREATE TABLE roles(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, persona_prompt TEXT, description TEXT);
            INSERT INTO roles(name, persona_prompt, description) VALUES('医生', 'mock', 'mock doctor');
            """
        )
        raw.execute(
            "INSERT INTO users(id, username, password_hash) VALUES(1, ?, ?)",
            ("existing", single.password_hash("pass1234")),
        )
        raw.commit()
        raw.close()
        self.database_patch = patch.object(single, "database", side_effect=lambda: SQLiteConnectionAdapter(self.db_path))
        self.database_patch.start()
        self.integrity_patch = patch.object(single.mysql.connector, "IntegrityError", sqlite3.IntegrityError)
        self.integrity_patch.start()
        self.client = TestClient(single.app)

    def tearDown(self):
        single.engine = None
        self.integrity_patch.stop()
        self.database_patch.stop()
        self.tempdir.cleanup()

    def test_register_login_duplicate_and_wrong_password(self):
        created = self.client.post("/api/register", json={"username": "alice", "password": "pass1234"})
        self.assertEqual(200, created.status_code)
        self.assertGreater(created.json()["user_id"], 0)

        duplicate = self.client.post("/api/register", json={"username": "alice", "password": "other123"})
        self.assertEqual(400, duplicate.status_code)
        self.assertEqual("用户名已存在", duplicate.json()["detail"])

        logged_in = self.client.post("/api/login", json={"username": "alice", "password": "pass1234"})
        self.assertEqual(200, logged_in.status_code)
        self.assertEqual("alice", logged_in.json()["username"])
        denied = self.client.post("/api/login", json={"username": "alice", "password": "wrong"})
        self.assertEqual(401, denied.status_code)

    def test_model_errors_are_mapped(self):
        request = httpx.Request("POST", "https://mock.invalid/v1/chat/completions")
        cases = [
            (APITimeoutError(request=request), 504),
            (APIConnectionError(request=request), 503),
            (OpenAIError("bad model response"), 502),
            (RuntimeError("milvus unavailable"), 503),
        ]
        for error, expected in cases:
            fake_engine = Mock()
            fake_engine.chat.side_effect = error
            with self.subTest(error=type(error).__name__), patch.object(single, "get_engine", return_value=fake_engine):
                response = self.client.post("/api/chat", json={"user_id": 1, "role_id": 1, "message": "高血压"})
                self.assertEqual(expected, response.status_code)

    def test_chat_rejects_unknown_user_before_engine_load(self):
        with patch.object(single, "get_engine") as get_engine:
            response = self.client.post(
                "/api/chat", json={"user_id": 999, "role_id": 1, "message": "高血压"}
            )
        self.assertEqual(404, response.status_code)
        self.assertEqual("用户不存在，请先注册登录", response.json()["detail"])
        get_engine.assert_not_called()

    def test_status_false_when_any_service_check_fails(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"id": "mock-model"}]}
        with patch.object(single.httpx, "get", return_value=response), patch.object(
            single, "MilvusClient", side_effect=RuntimeError("offline")
        ), patch.dict(os.environ, {
            "LLM_API_KEY": "mock-key",
            "LLM_BASE_URL": "https://mock.invalid/v1",
            "LLM_MODEL": "mock-model",
        }):
            result = self.client.get("/api/status")
        self.assertEqual(200, result.status_code)
        self.assertTrue(result.json()["model_ready"])
        self.assertFalse(result.json()["knowledge_ready"])
        self.assertFalse(result.json()["ready"])

    def test_whitespace_only_message_is_rejected_before_engine(self):
        fake_engine = Mock()
        fake_engine.chat.return_value = {"answer": "mock", "sources": [], "rewritten_query": "", "rerank_state": "disabled"}
        with patch.object(single, "get_engine", return_value=fake_engine):
            response = self.client.post("/api/chat", json={"user_id": 1, "role_id": 1, "message": "   "})
        self.assertEqual(422, response.status_code)
        fake_engine.chat.assert_not_called()

    def test_question_numeric_and_length_boundaries(self):
        payloads = [
            {"user_id": 0, "role_id": 1, "message": "高血压"},
            {"user_id": 1, "role_id": 0, "message": "高血压"},
            {"user_id": 1, "role_id": 1, "message": "高" * 501},
        ]
        with patch.object(single, "get_engine") as get_engine:
            for payload in payloads:
                with self.subTest(payload=payload):
                    self.assertEqual(422, self.client.post("/api/chat", json=payload).status_code)
        get_engine.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
