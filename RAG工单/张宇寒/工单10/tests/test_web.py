import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

import app as web


ROOT = Path(__file__).resolve().parents[1]


def run_immediate(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("本地证据模式不应等待网络")


class FakeGraphRAG:
    def search(self, question, top_k=5):
        return {
            "hits": [{"chunk_id": "c1", "document": "平安银行2019年年报", "stock_code": "000001", "company": "平安银行", "year": 2019, "page": 5, "text": "营业收入1,379.58亿元。", "title": "摘要", "rank": 1, "score": 0.9}],
            "graph": {"nodes": [{"id": "chunk:c1", "label": "平安银行 第5页", "type": "chunk", "metadata": {}}], "edges": []},
            "intent": {"stocks": ["000001"], "years": [2019], "metrics": ["营业收入"]},
            "retrieval_ms": 12.5,
        }


class WebBehaviorTests(unittest.TestCase):
    def test_questions_endpoint_reads_the_fixed_ten_questions(self):
        payload = web.question_list()
        self.assertEqual(payload["count"], 10)
        self.assertEqual(payload["questions"][0]["id"], 1)
        self.assertIn("Question", (ROOT / "eval_questions.md").read_text(encoding="utf-8"))

    def test_local_evidence_mode_returns_hits_and_graph_without_api_key(self):
        request = web.AskRequest(question="平安银行营业收入是多少？", top_k=3)
        with patch.object(web, "get_rag", return_value=FakeGraphRAG()), patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            payload = run_immediate(web.ask(request))
        self.assertEqual(payload["answer_mode"], "evidence")
        self.assertIn("1,379.58", payload["answer"])
        self.assertEqual(payload["graph"]["nodes"][0]["type"], "chunk")

    def test_request_rejects_empty_question_and_out_of_range_top_k(self):
        with self.assertRaises(ValidationError):
            web.AskRequest(question="", top_k=5)
        with self.assertRaises(ValidationError):
            web.AskRequest(question="收入", top_k=11)

    def test_health_explains_missing_index(self):
        with patch.object(web, "DATA_DIR", ROOT / "not-created"):
            payload = web.health()
        self.assertFalse(payload["ready"])
        self.assertIn("prepare.py", payload["message"])

    def test_run_server_uses_port_8021(self):
        with patch("uvicorn.run") as runner:
            web.run_server()
        runner.assert_called_once_with("app:app", host="127.0.0.1", port=8021, reload=False)

    def test_page_has_question_answer_sources_and_svg_graph_regions(self):
        page = (ROOT / "index.html").read_text(encoding="utf-8")
        for marker in ('id="question"', 'id="answer"', 'id="baselineSources"', 'id="optimizedSources"', 'id="graphCanvas"', '/api/evaluate', '/api/benchmarks'):
            self.assertIn(marker, page)


if __name__ == "__main__":
    unittest.main()

