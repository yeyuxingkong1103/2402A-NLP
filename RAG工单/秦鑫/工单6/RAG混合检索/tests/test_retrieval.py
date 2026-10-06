import unittest

import server


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.previous = server._chunks
        server._chunks = [
            {"id": "a", "document": "招股说明书.pdf", "page": 25, "text": "公司注册资本为人民币七千万元，法定代表人为张某。"},
            {"id": "b", "document": "招股说明书.pdf", "page": 80, "text": "报告期内公司军用领域营业收入为一万万元。"},
            {"id": "c", "document": "其他.pdf", "page": 1, "text": "本合同约定付款期限为三十日。"},
        ]

    def tearDown(self):
        server._chunks = self.previous

    def test_chinese_character_ngrams_support_retrieval(self):
        results, metrics = server.retrieve("公司的注册资金是多少？", top_k=2)
        self.assertEqual(results[0]["id"], "a")
        self.assertEqual(results[0]["page"], 25)
        self.assertIn("RRF", metrics["routes"])

    def test_query_expansion_adds_domain_terms(self):
        rewritten = server.expanded_query("军用领域收入")
        self.assertIn("营业收入", rewritten)

    def test_sentence_chunking_keeps_content_and_bounds_length(self):
        parts = server.split_page("第一句用于检索测试。" * 100, size=80, overlap=10)
        self.assertTrue(parts)
        self.assertTrue(all(len(part) <= 80 for part in parts))


if __name__ == "__main__":
    unittest.main()
