import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch


class HtmlIds(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, _tag, attrs):
        value = dict(attrs).get("id")
        if value:
            self.ids.add(value)


class FakeRAG:
    async def answer(self, question, top_k=5, mode="hybrid"):
        return {
            "answer": "证据摘要",
            "mode": "evidence",
            "retrieval_ms": 1.0,
            "generation_ms": 0.0,
            "total_ms": 1.0,
            "hits": [{"stock_code": "000001", "page": 5, "text": "证据", "score": 0.9}],
        }


class WebTests(unittest.TestCase):
    def test_app_import_does_not_emit_swig_deprecation_warnings(self):
        result = subprocess.run(
            [sys.executable, "-W", "default", "-c", "import app"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        self.assertEqual(result.returncode, 0)
        self.assertNotIn("SwigPy", result.stderr)

    def test_questions_endpoint_returns_fixed_ten(self):
        from app import evaluation_questions

        result = evaluation_questions()

        self.assertEqual(result["count"], 10)
        self.assertEqual(len(result["questions"]), 10)

    def test_run_endpoint_writes_downloadable_json_report(self):
        from app import EvaluationRequest, run_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            with patch("app.get_rag", return_value=FakeRAG()), patch("app.REPORT_PATH", report_path):
                result = asyncio.run(run_evaluation(EvaluationRequest(top_k=5, mode="hybrid")))

            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result["summary"]["total"], 10)
        self.assertEqual(saved["summary"], result["summary"])

    def test_health_explains_missing_index(self):
        from app import health

        with tempfile.TemporaryDirectory() as tmp, patch("app.DATA_DIR", Path(tmp)):
            result = health()

        self.assertFalse(result["index_ready"])
        self.assertIn("prepare.py", result["message"])

    def test_page_has_evaluation_controls_and_result_regions(self):
        page = (Path(__file__).resolve().parents[1] / "index.html").read_text(encoding="utf-8")
        parser = HtmlIds()
        parser.feed(page)
        expected = {"runEvaluation", "downloadReport", "retrievalMode", "topK", "summary", "results", "status"}
        self.assertTrue(expected.issubset(parser.ids), expected - parser.ids)
        self.assertIn("固定10题", page)


if __name__ == "__main__":
    unittest.main()
