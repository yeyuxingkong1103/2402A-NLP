import asyncio
import os
import unittest
from unittest.mock import patch


class RetrievalIntentTests(unittest.TestCase):
    def test_extracts_one_or_multiple_named_companies(self):
        from retrieval import extract_stock_codes

        self.assertEqual(extract_stock_codes("平安银行2019年营业收入是多少？"), ["000001"])
        self.assertEqual(
            extract_stock_codes("比较中信证券、招商证券和国泰君安的营业收入"),
            ["600030", "600999", "601211"],
        )

    def test_report_aware_selection_filters_single_company(self):
        from retrieval import select_candidate_indices

        chunks = [
            type("C", (), {"stock_code": "601318", "page": 10, "text": "营业收入 净利润"})(),
            type("C", (), {"stock_code": "000001", "page": 5, "text": "营业收入 净利润"})(),
        ]

        order = select_candidate_indices("平安银行营业收入和净利润", chunks, [0.95, 0.70], top_k=5)

        self.assertEqual(order, [1])

    def test_multi_company_selection_keeps_evidence_from_each_company(self):
        from retrieval import select_candidate_indices

        chunks = [
            type("C", (), {"stock_code": "600030", "page": 17, "text": "营业收入543.83亿元"})(),
            type("C", (), {"stock_code": "600030", "page": 30, "text": "营业收入"})(),
            type("C", (), {"stock_code": "600999", "page": 34, "text": "营业收入294.29亿元"})(),
            type("C", (), {"stock_code": "601211", "page": 29, "text": "营业收入428.17亿元"})(),
        ]

        order = select_candidate_indices(
            "比较中信证券、招商证券和国泰君安的营业收入", chunks, [0.9, 0.8, 0.7, 0.6], top_k=3
        )

        self.assertEqual({chunks[index].stock_code for index in order}, {"600030", "600999", "601211"})

    def test_summary_page_with_all_requested_metrics_is_promoted(self):
        from retrieval import select_candidate_indices

        chunks = [
            type("C", (), {"stock_code": "000001", "page": 135, "text": "营业收入明细"})(),
            type("C", (), {"stock_code": "000001", "page": 5, "text": "实现营业收入1379.58亿元，净利润281.95亿元"})(),
        ]

        order = select_candidate_indices("平安银行营业收入和净利润分别是多少", chunks, [0.56, 0.51], top_k=2)

        self.assertEqual(order[0], 1)

    def test_evidence_fallback_keeps_all_top_k_hits(self):
        from retrieval import RAGSystem

        class FakeIndex:
            def search(self, question, top_k=5, mode="hybrid"):
                return [
                    {"document": "报告", "stock_code": "000001", "page": index, "text": f"证据{index}", "score": 1.0}
                    for index in range(1, 6)
                ]

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            result = asyncio.run(RAGSystem(FakeIndex()).answer("问题", top_k=5))

        self.assertIn("证据5", result["answer"])


if __name__ == "__main__":
    unittest.main()
