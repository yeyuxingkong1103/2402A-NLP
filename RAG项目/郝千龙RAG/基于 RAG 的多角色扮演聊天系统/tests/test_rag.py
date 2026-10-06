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
        """类级夹具：所有用例共享同一份迷你句对与检索器，BM25 索引只构建一次，避免重复开销。"""
        # 类级初始化：所有用例共用一份检索器（索引只建一次）
        cls.pairs = load_pairs(MINI)
        cls.retriever = TranslationRetriever.build(MINI)

    def test_load_pairs(self):
        """验证 TSV 句对加载器：条数达到预期且首条英文内容解析正确。"""
        # 数据加载：条数正确、首条英文匹配
        self.assertGreaterEqual(len(self.pairs), 8)
        self.assertEqual(self.pairs[0].english, "I have to go to sleep.")

    def test_tokenize_bilingual(self):
        """验证中英混合分词：英文按词切分、中文走 jieba，两种 token 都应出现在结果中。"""
        # 双语分词：英文 token 与中文 token 都应存在
        tokens = tokenize("I miss you 我想你")
        self.assertTrue(any(t.lower() == "miss" for t in tokens))
        self.assertTrue(any("想" in t or t == "想" for t in tokens) or "我想你" in "".join(tokens))

    def test_search_chinese(self):
        """验证中文查询「我想你」能跨语言召回含 miss 的英文句对（语义/对译对齐召回）。"""
        # 中文查询应命中含 miss 的英文句
        hits = self.retriever.search("我想你", top_k=3)
        self.assertTrue(hits)
        joined = " ".join(p.english.lower() for p, _ in hits)
        self.assertIn("miss", joined)

    def test_search_english(self):
        """验证英文查询「Hurry up」能召回含 hurry 的相关句对（英文关键词精确命中）。"""
        # 英文查询应命中含 hurry 的句子
        hits = self.retriever.search("Hurry up", top_k=3)
        self.assertTrue(hits)
        joined = " ".join(p.english.lower() for p, _ in hits)
        self.assertIn("hurry", joined)

    def test_hybrid_search(self):
        """验证混合检索主入口：无论向量路是否启用，hybrid_search 都应降级保底返回非空结果。"""
        # 混合检索（BM25 + 可选向量）至少应返回结果
        hits = self.retriever.hybrid_search("go to sleep", top_k=3)
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
