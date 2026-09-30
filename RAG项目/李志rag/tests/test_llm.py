import unittest
from unittest.mock import MagicMock, patch

from app.rag_main import MEDICAL_NOTICE, build_generation_messages, generate_answer, postprocess


class LlmTests(unittest.TestCase):
    def test_generation_prompt_contains_role_history_and_evidence(self):
        messages = build_generation_messages(
            "控制目标？", "你是医疗助手", [{"role": "assistant", "content": "历史"}],
            [{"source": "指南.pdf", "text": "目标低于140/90"}],
        )
        self.assertIn("你是医疗助手", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "历史")
        self.assertIn("指南.pdf", messages[-1]["content"])

    def test_postprocess_removes_reasoning_tag(self):
        self.assertEqual(postprocess("<think>隐藏</think>\n答案"), "答案")

    def test_emergency_short_circuits_model(self):
        with patch("app.rag_main.httpx.Client") as client:
            answer = generate_answer("我胸痛而且呼吸困难", "提示", [], [])
        client.assert_not_called()
        self.assertIn("120", answer)
        self.assertIn(MEDICAL_NOTICE, answer)

    def test_general_knowledge_fallback_without_hits(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "通用知识回答"}}]
        }
        client = MagicMock()
        client.__enter__.return_value.post.return_value = response
        with patch("app.rag_main.httpx.Client", return_value=client):
            answer = generate_answer("知识库之外的问题", "提示", [], [])
        request = client.__enter__.return_value.post.call_args.kwargs["json"]
        self.assertIn("可以使用可靠的通用知识回答", request["messages"][0]["content"])
        self.assertIn("通用知识回答", answer)


if __name__ == "__main__":
    unittest.main()
