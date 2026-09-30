import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rag_answer


class RagAnswerTests(unittest.TestCase):
    def test_cli_defaults_to_deepseek_chat(self):
        args = rag_answer.parse_args([])

        self.assertEqual(args.model, "deepseek-chat")
        self.assertEqual(args.timeout, 60)

    def test_cli_accepts_model_and_timeout(self):
        args = rag_answer.parse_args(["--model", "deepseek-reasoner", "--timeout", "120"])

        self.assertEqual(args.model, "deepseek-reasoner")
        self.assertEqual(args.timeout, 120)

    def test_call_deepseek_api_uses_env_key_and_request_options(self):
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "回答 [章节]"})()})()]},
                )()

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        with patch.dict(rag_answer.os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            with patch.object(rag_answer, "OpenAI", FakeClient):
                answer = rag_answer.call_deepseek_api("提示词", model="deepseek-chat", timeout=60)

        self.assertEqual(answer, "回答 [章节]")
        self.assertEqual(captured["client"], {"api_key": "test-key", "base_url": "https://api.deepseek.com"})
        self.assertEqual(captured["model"], "deepseek-chat")
        self.assertEqual(captured["messages"], [{"role": "user", "content": "提示词"}])
        self.assertEqual(captured["temperature"], 0.1)
        self.assertFalse(captured["stream"])
        self.assertEqual(captured["timeout"], 60)

    def test_build_context_marks_each_chunk_source(self):
        context = rag_answer.build_retrieved_context(
            [
                {
                    "section_path": "5 高血压治疗 > 药物治疗",
                    "text": "名称: 氨氯地平 | 禁忌证: 快速性心律失常",
                }
            ]
        )

        self.assertIn("[来源: 5 高血压治疗 > 药物治疗]", context)
        self.assertIn("名称: 氨氯地平", context)

    def test_build_prompt_contains_required_template(self):
        prompt = rag_answer.build_prompt("问题", "上下文")

        self.assertIn("你是基层高血压管理助手。请仅基于以下上下文回答问题。", prompt)
        self.assertIn(
            "必须在回答的最后一行，用 [章节路径] 格式列出所有引用的来源。这是强制要求，不能不写。",
            prompt,
        )
        self.assertIn("根据现有指南无法回答该问题", prompt)
        self.assertIn("【上下文】\n上下文", prompt)
        self.assertIn("【问题】\n问题", prompt)

    def test_build_prompt_contains_memory_context_without_replacing_guideline_context(self):
        prompt = rag_answer.build_prompt(
            "那运动方面呢？",
            "[来源: 5 高血压治疗]\n指南内容",
            conversation_context="用户：我刚才问了降压药\n助手：请结合指南选择",
            long_term_memory="过敏史：对青霉素过敏",
        )

        self.assertIn("【最近对话】", prompt)
        self.assertIn("【用户已确认的长期记忆】", prompt)
        self.assertIn("过敏史：对青霉素过敏", prompt)
        self.assertIn("【上下文】\n[来源: 5 高血压治疗]\n指南内容", prompt)
        self.assertIn("长期记忆和最近对话不能替代指南上下文", prompt)

    def test_answer_question_uses_llm_fallback_when_no_chunks_retrieved(self):
        with patch.object(rag_answer, "retrieve_top5", return_value=[]):
            with patch.object(rag_answer, "call_deepseek_api", return_value="这是通用建议。") as mock_call:
                result = rag_answer.answer_question(
                    client="client",
                    collection="collection",
                    embedding_model="embedding",
                    reranker="reranker",
                    question="数据库没有的问题",
                    model="deepseek-chat",
                    timeout=60,
                )

        prompt = mock_call.call_args.kwargs.get("prompt") or mock_call.call_args.args[0]
        self.assertIn("未检索到本地知识库", prompt)
        self.assertEqual(result["answer_mode"], "llm_fallback")
        self.assertEqual(result["retrieved"], [])
        self.assertFalse(result["has_citation"])
        self.assertIn("不是基于数据库", result["notice"])
        self.assertIn("不是基于数据库", result["answer"])

    def test_answer_question_uses_fallback_when_chunks_are_irrelevant(self):
        fake_chunks = [
            {
                "section_path": "5 高血压治疗",
                "chunk_type": "text",
                "rerank_score": 0.18,
                "text": "高血压患者应规范管理。",
            }
        ]
        with patch.object(rag_answer, "retrieve_top5", return_value=fake_chunks):
            with patch.object(rag_answer, "call_deepseek_api", return_value="无法获取实时天气，请提供所在城市。") as mock_call:
                result = rag_answer.answer_question(
                    client="client",
                    collection="collection",
                    embedding_model="embedding",
                    reranker="reranker",
                    question="今天天气如何？",
                    model="deepseek-chat",
                    timeout=60,
                )

        prompt = mock_call.call_args.kwargs.get("prompt") or mock_call.call_args.args[0]
        self.assertIn("未检索到本地知识库", prompt)
        self.assertIn("实时信息", prompt)
        self.assertEqual(result["answer_mode"], "llm_fallback")
        self.assertEqual(result["retrieved"], [])
        self.assertFalse(result["has_citation"])

    def test_answer_question_marks_rag_mode_when_chunks_exist(self):
        fake_chunks = [
            {
                "section_path": "5 高血压治疗",
                "chunk_type": "text",
                "rerank_score": 0.9,
                "text": "指南内容",
            }
        ]
        with patch.object(rag_answer, "retrieve_top5", return_value=fake_chunks):
            with patch.object(rag_answer, "call_deepseek_api", return_value="回答 [5 高血压治疗]"):
                result = rag_answer.answer_question(
                    client="client",
                    collection="collection",
                    embedding_model="embedding",
                    reranker="reranker",
                    question="指南内问题",
                    model="deepseek-chat",
                    timeout=60,
                )

        self.assertEqual(result["answer_mode"], "rag")
        self.assertIn("本地知识库", result["notice"])

        fake_chunks = [
            {
                "section_path": "5 高血压治疗",
                "chunk_type": "text",
                "rerank_score": 0.9,
                "text": "指南内容",
            }
        ]
        with patch.object(rag_answer, "retrieve_top5", return_value=fake_chunks):
            with patch.object(rag_answer, "call_deepseek_api", return_value="回答 [5 高血压治疗]") as mock_call:
                rag_answer.answer_question(
                    client="client",
                    collection="collection",
                    embedding_model="embedding",
                    reranker="reranker",
                    question="那运动方面呢？",
                    model="deepseek-chat",
                    timeout=60,
                    conversation_context="用户：我刚才问了降压药",
                    long_term_memory="偏好：回答简洁",
                )

        prompt = mock_call.call_args.kwargs.get("prompt") or mock_call.call_args.args[0]
        self.assertIn("我刚才问了降压药", prompt)
        self.assertIn("偏好：回答简洁", prompt)

    def test_analyze_answer_detects_citation_and_unanswerable(self):
        self.assertTrue(rag_answer.has_citation("结论。[5 高血压治疗]"))
        self.assertFalse(rag_answer.has_citation("结论，无引用。"))
        self.assertTrue(rag_answer.is_unanswerable("根据现有指南无法回答该问题"))

    def test_analyze_answer_detects_citation_and_unanswerable(self):
        self.assertTrue(rag_answer.has_citation("结论。[5 高血压治疗]"))
        self.assertFalse(rag_answer.has_citation("结论，无引用。"))
        self.assertTrue(rag_answer.is_unanswerable("根据现有指南无法回答该问题"))

    def test_strip_r1_reasoning_keeps_final_answer(self):
        answer = "思考：让我分析一下。首先查看上下文。\n最终答案：应选用 ACEI。[药物治疗]"

        cleaned = rag_answer.strip_r1_reasoning(answer)

        self.assertEqual(cleaned, "应选用 ACEI。[药物治疗]")

    def test_ensure_citation_auto_appends_top1_source(self):
        answer, status = rag_answer.ensure_citation("应选用 ACEI。", "5 高血压治疗")

        self.assertEqual(status, "auto_appended")
        self.assertTrue(answer.endswith("（参考来源：5 高血压治疗）"))

    def test_format_retrieved_keeps_top5_metadata(self):
        retrieved = rag_answer.format_retrieved(
            [
                {
                    "section_path": "章节",
                    "chunk_type": "table_row",
                    "rerank_score": 0.9,
                    "text": "正文" * 80,
                }
            ]
        )

        self.assertEqual(retrieved[0]["section_path"], "章节")
        self.assertEqual(retrieved[0]["chunk_type"], "table_row")
        self.assertEqual(retrieved[0]["rerank_score"], 0.9)
        self.assertLessEqual(len(retrieved[0]["text_preview"]), 100)


if __name__ == "__main__":
    unittest.main()
