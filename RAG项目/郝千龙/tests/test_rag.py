# -*- coding: utf-8 -*-
"""检索单元测试：数据加载 / 分词 / 中英文检索 / 混合检索。"""
import unittest
from pathlib import Path

from rag import TranslationRetriever, load_pairs, tokenize

# 测试夹具：10 条迷你中英句对，避免依赖外部大数据集
MINI = Path(__file__).resolve().parent / "fixtures" / "mini.tsv"


class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 类级初始化：所有用例共用一份检索器（索引只建一次）
        cls.pairs = load_pairs(MINI)
        cls.retriever = TranslationRetriever.build(MINI)

    def test_load_pairs(self):
        # 数据加载：条数正确、首条英文匹配
        self.assertGreaterEqual(len(self.pairs), 8)
        self.assertEqual(self.pairs[0].english, "I have to go to sleep.")

    def test_tokenize_bilingual(self):
        # 双语分词：英文 token 与中文 token 都应存在
        tokens = tokenize("I miss you 我想你")
        self.assertTrue(any(t.lower() == "miss" for t in tokens))
        self.assertTrue(any("想" in t or t == "想" for t in tokens) or "我想你" in "".join(tokens))

    def test_search_chinese(self):
        # 中文查询应命中含 miss 的英文句
        hits = self.retriever.search("我想你", top_k=3)
        self.assertTrue(hits)
        joined = " ".join(p.english.lower() for p, _ in hits)
        self.assertIn("miss", joined)

    def test_search_english(self):
        # 英文查询应命中含 hurry 的句子
        hits = self.retriever.search("Hurry up", top_k=3)
        self.assertTrue(hits)
        joined = " ".join(p.english.lower() for p, _ in hits)
        self.assertIn("hurry", joined)

    def test_hybrid_search(self):
        # 混合检索（BM25 + 可选向量）至少应返回结果
        hits = self.retriever.hybrid_search("go to sleep", top_k=3)
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
