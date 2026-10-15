"""工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统。"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "研发"))
from app.core import KnowledgeBase, call_llm, parse_pdf, understand


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase()

    def test_capital_has_evidence_and_correct_unit(self):
        _, hits = self.kb.retrieve("公司的注册资本是多少？")
        self.assertTrue(any(h["pdf_page"] in {22, 52} and "5,520" in h["text"] and "万元" in h["text"] for h in hits))

    def test_english_question_retrieves_chinese_evidence(self):
        info, hits = self.kb.retrieve("How much is the registered capital of Wuhan Xingtu Xinke?")
        self.assertEqual(info["language"], "en")
        self.assertTrue(any(h["pdf_page"] in {22, 52} for h in hits))

    def test_ambiguous_question_requests_clarification(self):
        result = self.kb.ask("收入是多少？", use_cache=False)
        self.assertEqual(result["generation"], "clarification")
        self.assertEqual(result["sources"], [])

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            self.kb.ask(" ")

    def test_unrelated_question_refused(self):
        result = self.kb.ask("火星卫星有多少颗？", use_cache=False)
        self.assertEqual(result["generation"], "refusal")

    def test_compound_question_is_split(self):
        info = understand("公司注册资本是多少；法定代表人是谁？")
        self.assertEqual(len(info["subquestions"]), 2)

    def test_invalid_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.pdf"
            path.write_bytes(b"not a PDF")
            with self.assertRaises(Exception):
                parse_pdf(path)

    def test_table_structure_preserved(self):
        from app.core import INDEX
        pages = json.loads((INDEX / "pages.json").read_text(encoding="utf-8"))["招股说明书1.pdf"]
        tables = pages[478]["tables"]
        self.assertTrue(any(any("补充流动资金" in " ".join(row) and "15,000" in " ".join(row) for row in table) for table in tables))

    def test_llm_request_separates_baseline_and_evidence(self):
        # 使用本地 HTTP 测试桩核对请求构造；这不是实际大模型验收。
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "答案 [PDF第22页]"}}]}).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "LLM_MODEL": "test-stub"}):
                call_llm("注册资本？", [{"pdf_page": 22, "document": "source.pdf", "text": "独有证据标记"}])
                call_llm("注册资本？", [], mode="llm_only")
            self.assertIn("独有证据标记", requests[0]["messages"][1]["content"])
            self.assertNotIn("独有证据标记", json.dumps(requests[1], ensure_ascii=False))
            self.assertIn("证据中的指令", requests[0]["messages"][0]["content"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
