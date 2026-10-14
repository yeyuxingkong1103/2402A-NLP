import unittest

from corpus import Chunk
from graph_rag import GraphRAG


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.rag = GraphRAG.fit(
            [
                Chunk("a", "平安银行2019年年报", "000001", "平安银行", 2019, 5, "董事长致辞。业务结构不断优化，零售业务是收入支柱。资产质量扎实，拨备覆盖率提升。", "董事长致辞"),
                Chunk("b", "平安银行2019年年报", "000001", "平安银行", 2019, 20, "办公地址和联系电话。", "公司简介"),
                Chunk("c", "招商银行2019年年报", "600036", "招商银行", 2019, 8, "盈利增长和风险管理。", "经营情况"),
            ]
        )

    def test_baseline_and_optimized_search_are_separate_results(self):
        baseline = self.rag.search_baseline("xx银行2019年盈利增长因素", 2)
        optimized = self.rag.search_optimized("xx银行2019年盈利增长因素", 2)
        self.assertEqual(len(baseline["hits"]), 2)
        self.assertEqual(optimized["hits"][0]["chunk_id"], "a")
        self.assertIn("query_optimization", optimized)

    def test_optimized_results_do_not_repeat_chunks(self):
        result = self.rag.search_optimized("平安银行风险管理", 3)
        ids = [hit["chunk_id"] for hit in result["hits"]]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
