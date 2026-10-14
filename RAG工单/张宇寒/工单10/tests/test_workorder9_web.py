import asyncio
from pathlib import Path
import unittest

import app as web


def run(coroutine):
    return asyncio.run(coroutine)


class FakeRAG:
    def search_baseline(self, question, top_k):
        return {"hits": [{"text": "无关内容", "chunk_id": "x"}], "retrieval_ms": 1.0}

    def search_optimized(self, question, top_k, preferred_query=""):
        hit = {"text": "业务结构优化，零售业务是收入支柱，拨备覆盖率不断提升。", "chunk_id": "a", "document": "平安银行2019年年报", "page": 5}
        return {"hits": [hit], "graph": {"nodes": [], "edges": []}, "intent": {"stocks": ["000001"], "years": [2019], "metrics": []}, "retrieval_ms": 2.0, "query_optimization": {"rewritten_query": "平安银行 业务结构 资产质量"}}


class Workorder9WebTests(unittest.TestCase):
    def setUp(self):
        self.old = web._rag
        web._rag = FakeRAG()

    def tearDown(self):
        web._rag = self.old

    def test_evaluate_returns_before_after_metrics_and_reference(self):
        payload = web.EvaluateRequest(question_id=1, top_k=5)
        result = run(web.evaluate(payload))
        self.assertIn("reference", result)
        self.assertEqual(result["optimized"]["metrics"]["context_recall"], 1.0)
        self.assertIn("delta", result)
        self.assertLess(result["total_ms"], 3000)

    def test_custom_question_has_no_fake_reference_metrics(self):
        result = run(web.evaluate(web.EvaluateRequest(question="平安银行董事长是谁？", top_k=3)))
        self.assertIsNone(result["reference"])
        self.assertIsNone(result["baseline"]["metrics"])

    def test_server_uses_workorder10_port(self):
        self.assertEqual(web.PORT, 8021)

    def test_page_exposes_metric_comparison_and_two_evidence_lists(self):
        page = (Path(__file__).parents[1] / "index.html").read_text(encoding="utf-8")
        for marker in ("metricCards", "optimizationLayers", "baselineSources", "optimizedSources"):
            self.assertIn(f'id="{marker}"', page)


if __name__ == "__main__":
    unittest.main()
