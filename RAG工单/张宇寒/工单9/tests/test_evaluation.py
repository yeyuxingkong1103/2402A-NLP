from pathlib import Path
import tempfile
import unittest

from evaluation import evaluate_hits, load_benchmarks, rewrite_query


class EvaluationTests(unittest.TestCase):
    def test_benchmark_parser_keeps_question_reference_and_facts(self):
        content = """# Sample\n## 1\n- Question: xx银行2019年的盈利因素？\n- Reference: 业务结构优化。\n- Facts: 业务结构优化;零售业务\n- Stocks: 000001\n"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "questions.md"
            path.write_text(content, encoding="utf-8")
            rows = load_benchmarks(path)
        self.assertEqual(rows[0]["question"], "xx银行2019年的盈利因素？")
        self.assertEqual(rows[0]["facts"], [["业务结构优化"], ["零售业务"]])
        self.assertEqual(rows[0]["stocks"], ["000001"])

    def test_anonymous_bank_and_topic_are_rewritten_for_retrieval(self):
        rewritten, explanation = rewrite_query("xx银行2019年盈利增长的关键因素有哪些？")
        self.assertIn("平安银行", rewritten)
        self.assertIn("业务结构", rewritten)
        self.assertIn("资产质量", rewritten)
        self.assertTrue(explanation["entities"])
        self.assertTrue(explanation["relations"])

    def test_precision_and_recall_use_reference_facts(self):
        hits = [
            {"text": "业务结构优化，零售业务成为收入支柱，拨备覆盖率提升。"},
            {"text": "无关的办公地址信息。"},
        ]
        benchmark = {"facts": [["业务结构优化"], ["零售业务"], ["拨备覆盖率", "覆盖率提升"]]}
        metrics = evaluate_hits(hits, benchmark)
        self.assertEqual(metrics["context_precision"], 1.0)
        self.assertEqual(metrics["context_recall"], 1.0)
        self.assertEqual(metrics["covered_facts"], 3)

    def test_empty_hits_are_zero_not_an_error(self):
        metrics = evaluate_hits([], {"facts": [["风险管理"]]})
        self.assertEqual(metrics["context_precision"], 0.0)
        self.assertEqual(metrics["context_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
