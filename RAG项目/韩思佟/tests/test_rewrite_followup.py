"""追问改写的 Mock 单元测试；不会连接真实模型或数据服务。"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


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
_module_names = ("mysql", "mysql.connector", "pymilvus", "sentence_transformers")
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
    _mysql_connector = types.ModuleType("mysql.connector")
    _mysql_connector.connect = Mock()
    _mysql = types.ModuleType("mysql")
    _mysql.connector = _mysql_connector
    sys.modules["mysql"] = _mysql
    sys.modules["mysql.connector"] = _mysql_connector
    sys.modules["pymilvus"] = types.SimpleNamespace(MilvusClient=StubMilvusClient)
    sys.modules["sentence_transformers"] = types.SimpleNamespace(
        CrossEncoder=StubCrossEncoder,
        SentenceTransformer=StubSentenceTransformer,
    )
    try:
        spec = importlib.util.spec_from_file_location("single_app_rewrite_under_test", SOURCE)
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


class RewriteFollowupTests(unittest.TestCase):
    @staticmethod
    def rag_with_rewrite(text):
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        )
        create = Mock(return_value=response)
        rag = object.__new__(single.RAG)
        rag.llm = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )
        rag.model = "mock-model"
        rag.options = {}
        return rag, create

    def test_history_topic_is_restored_when_model_omits_it(self):
        rag, create = self.rag_with_rewrite("每天应该测几次？")

        with patch.dict(os.environ, {"RAG_KNOWLEDGE_SCOPE_TERMS": "高血压,血压"}):
            rewritten = rag.rewrite("用户：我的血压最近有点高", "每天应该测几次？")

        self.assertEqual("血压每天应该测几次？", rewritten)
        create.assert_called_once()

    def test_complete_topic_is_not_duplicated(self):
        rag, create = self.rag_with_rewrite("血压每天应该测几次？")

        with patch.dict(os.environ, {"RAG_KNOWLEDGE_SCOPE_TERMS": "高血压,血压"}):
            rewritten = rag.rewrite("用户：我的血压最近有点高", "每天应该测几次？")

        self.assertEqual("血压每天应该测几次？", rewritten)
        self.assertEqual(1, rewritten.count("血压"))
        create.assert_called_once()

    def test_first_turn_returns_original_question_without_api_call(self):
        rag, create = self.rag_with_rewrite("不应使用这个结果")

        rewritten = rag.rewrite("（无历史对话）", "血压每天应该测几次？")

        self.assertEqual("血压每天应该测几次？", rewritten)
        create.assert_not_called()

    def test_explanation_is_rejected_and_topic_is_preserved(self):
        rationale = "根据历史对话，用户问测量前应该安静休息多久，助手已经说过五分钟，可能需要确认。"
        rag, _ = self.rag_with_rewrite(rationale)
        with patch.dict(os.environ, {"RAG_KNOWLEDGE_SCOPE_TERMS": "血压"}):
            self.assertEqual(rag.rewrite("用户：在家测量血压", "那测量前应该安静休息多久？"),
                             "血压那测量前应该安静休息多久？")

    def test_multiline_or_truncated_analysis_does_not_become_query(self):
        for output in ("先分析历史\n血压测量前需要休息多久？", "测量" * 100, "<think>分析问题"):
            with self.subTest(output=output):
                rag, _ = self.rag_with_rewrite(output)
                with patch.dict(os.environ, {"RAG_KNOWLEDGE_SCOPE_TERMS": "血压"}):
                    self.assertEqual(rag.rewrite("用户：在家测量血压", "需要休息多久？"),
                                     "血压需要休息多久？")


if __name__ == "__main__":
    unittest.main()
