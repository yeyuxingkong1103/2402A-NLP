"""
test_unit.py — 单元测试（纯函数部分）

覆盖需求文档 4.14 要求的五类用例中的意图识别、分块函数、余弦相似度过滤与去重。
全部用例均可离线运行，不调用大模型、不依赖 Redis / Milvus：

    python -m unittest tests.test_unit -v
    python tests/test_unit.py

需要真实向量库与 PDF 解析的用例在 tests/test_storage.py。
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  （必须先执行，才能安全导入业务模块）

import chunking  # noqa: E402
import config  # noqa: E402
import domains  # noqa: E402
import embeddings  # noqa: E402
import intent  # noqa: E402
import long_term  # noqa: E402
import rerank_filter  # noqa: E402


class TestIntent(unittest.TestCase):
    """意图识别（关键词路径与标签解析，不依赖模型）。"""

    def test_keyword_detection(self):
        cases = {
            "劳动合同到期不续签，公司需要赔偿吗？": "legal",
            "我最近总是失眠，需要吃药吗？": "medical",
            "虚拟语气怎么用，能举个例子吗？": "english",
            "邻居装修太吵我能报警吗": "legal",
            "孩子发烧到39度怎么办": "medical",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(intent.detect_by_keywords(text), expected)

    def test_small_talk(self):
        for text in ("你好", "hi", "谢谢"):
            with self.subTest(text=text):
                self.assertEqual(intent.detect_by_keywords(text), "chat")

    def test_parse_label(self):
        self.assertEqual(intent.parse_label("legal"), "legal")
        self.assertEqual(intent.parse_label("**Medical**"), "medical")
        self.assertEqual(intent.parse_label("这属于法律问题"), "legal")
        self.assertIsNone(intent.parse_label("完全无关的内容"))

    def test_empty_input(self):
        self.assertEqual(intent.detect_intent(""), "chat")
        self.assertIsNone(intent.detect_by_keywords(""))

    def test_domain_to_role(self):
        self.assertEqual(domains.role_for("legal"), "法律顾问")
        self.assertEqual(domains.role_for("medical"), "医疗咨询")
        self.assertEqual(domains.role_for("english"), "英语学习助手")
        self.assertEqual(domains.role_for("chat"), domains.CHAT_ROLE_NAME)


class TestDomains(unittest.TestCase):
    """领域类：注册表、置信度打分、四段式提示词。"""

    def test_registry(self):
        self.assertEqual(
            [d.domain for d in domains.all_domains()], ["legal", "medical", "english"]
        )
        self.assertEqual(domains.get_domain("medical").collection, "kb_medical")
        self.assertEqual(domains.get_domain("不存在的领域").domain, "chat")
        self.assertEqual(len(domains.list_roles()), 3)

    def test_route_confidence(self):
        legal = domains.get_domain("legal")
        self.assertEqual(legal.route("劳动合同到期不续签，公司要赔偿吗？"), 1.0)
        self.assertEqual(legal.route("今天天气不错"), 0.0)
        self.assertEqual(legal.route(""), 0.0)

    def test_chat_domain_skips_retrieval(self):
        chat = domains.chat_domain()
        self.assertEqual(chat.domain, "chat")
        self.assertEqual(chat.retrieve("随便问问"), [])
        self.assertNotIn("【知识库检索】", chat.build_system_prompt())

    def test_prompt_has_all_sections(self):
        prompt = domains.get_domain("legal").build_system_prompt(
            docs=[
                {
                    "text": "劳动者提前三十日通知可以解除劳动合同",
                    "source": "劳动法常见问题.pdf",
                    "page": 3,
                    "score": 0.9,
                }
            ],
            memory_text="- [2026-09-01 摘要] 用户是法学生",
            history=[{"role": "user", "content": "试用期一般多久"}],
        )
        for section in ("【历史对话参考】", "【知识库检索】", "【最近对话】"):
            self.assertIn(section, prompt)
        # 引用要拼成"文件名 第N页"
        self.assertIn("劳动法常见问题.pdf 第3页", prompt)
        self.assertIn("用户是法学生", prompt)

    def test_prompt_renders_empty_memory(self):
        # 长期记忆没召回时这一段也必须出现，不能整段消失
        prompt = domains.get_domain("medical").build_system_prompt()
        self.assertIn("【历史对话参考】", prompt)
        self.assertIn("暂无历史记忆", prompt)
        self.assertIn("本轮未检索到", prompt)


class TestLongTermConfig(unittest.TestCase):
    """长期记忆阈值按向量后端区分，时间衰减只影响排序。"""

    def test_threshold_follows_backend(self):
        original = config.EMBEDDING_BACKEND
        try:
            config.EMBEDDING_BACKEND = "hash"
            self.assertAlmostEqual(
                config.long_term_threshold(), config.LONG_TERM_THRESHOLD_HASH
            )
            config.EMBEDDING_BACKEND = "bge-m3"
            self.assertAlmostEqual(
                config.long_term_threshold(), config.LONG_TERM_THRESHOLD_BGE
            )
        finally:
            config.EMBEDDING_BACKEND = original

    def test_decay_prefers_recent(self):
        now = int(time.time())
        old = {"score": 0.8, "created_at": now - 60 * 86400}
        new = {"score": 0.75, "created_at": now}
        self.assertLess(long_term._decay_score(old), long_term._decay_score(new))

    def test_decay_leaves_raw_score_alone(self):
        # 衰减分可以低到阈值以下，但阈值过滤用的是原始余弦
        hit = {"score": 0.72, "created_at": int(time.time()) - 365 * 86400}
        self.assertLess(long_term._decay_score(hit), config.LONG_TERM_THRESHOLD_BGE)
        self.assertGreaterEqual(hit["score"], config.LONG_TERM_THRESHOLD_BGE)


class TestChunking(unittest.TestCase):
    """六种分块策略。"""

    SAMPLE = (
        "第一章 总则\n"
        "第一条 为了保护劳动者的合法权益，根据宪法，制定本法。\n"
        "第二条 在中华人民共和国境内的企业、个体经济组织适用本法。\n\n"
        "第二章 劳动合同的订立\n"
        "第十条 建立劳动关系，应当订立书面劳动合同。已建立劳动关系，"
        "未同时订立书面劳动合同的，应当自用工之日起一个月内订立书面劳动合同。\n"
    )

    def setUp(self):
        self.chunker = chunking.Chunker(size=80, overlap=10)

    def test_fixed(self):
        chunks = self.chunker.chunk_fixed(self.SAMPLE, size=50, overlap=5)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["text"]) <= 50 for c in chunks))

    def test_sentence_and_paragraph(self):
        for name in ("sentence", "paragraph"):
            self.assertGreater(len(self.chunker.chunk(self.SAMPLE, name)), 0, name)

    def test_heading_keeps_context(self):
        chunks = self.chunker.chunk(self.SAMPLE, "heading")
        self.assertTrue(chunks and any(c.get("heading") for c in chunks))

    def test_parent_child(self):
        chunks = self.chunker.chunk(self.SAMPLE, "parent_child")
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertIsNotNone(chunk.get("parent_index"))
            self.assertTrue(chunk.get("parent_text"))

    def test_no_empty_chunk(self):
        strategies = ("fixed", "sentence", "paragraph", "heading", "parent_child", "semantic")
        for name in strategies:
            with self.subTest(strategy=name):
                chunks = self.chunker.chunk(self.SAMPLE, name)
                self.assertTrue(all(c["text"].strip() for c in chunks))
                self.assertTrue(all(len(c["text"]) >= config.MIN_CHUNK_LENGTH for c in chunks))

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            self.chunker.chunk(self.SAMPLE, "not-a-strategy")


class TestEmbeddings(unittest.TestCase):
    """向量化的维度、归一化与稳定性。"""

    def test_dimension_and_normalize(self):
        import numpy as np

        vectors = embeddings.encode_texts(["劳动合同解除", "今天天气不错"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), config.EMBEDDING_DIM)
        for vector in vectors:
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=4)

    def test_deterministic(self):
        first = embeddings.encode_query("经济补偿金的计算标准")
        second = embeddings.encode_query("经济补偿金的计算标准")
        self.assertEqual(first, second)

    def test_empty_input(self):
        self.assertEqual(embeddings.encode_texts([]), [])


class TestRerankFilter(unittest.TestCase):
    """余弦相似度计算、阈值过滤与去重。"""

    @classmethod
    def setUpClass(cls):
        cls.vectors = embeddings.encode_texts(
            ["劳动合同如何解除", "劳动合同如何解除的说明", "今天天气真好"]
        )

    def test_cosine_range(self):
        same = rerank_filter.cosine_similarity(self.vectors[0], self.vectors[0])
        self.assertAlmostEqual(same, 1.0, places=4)
        self.assertLess(rerank_filter.cosine_similarity(self.vectors[0], self.vectors[2]), same)

    def test_cosine_invalid_input(self):
        self.assertEqual(rerank_filter.cosine_similarity(None, self.vectors[0]), 0.0)
        self.assertEqual(rerank_filter.cosine_similarity([0.0] * 8, [0.0] * 8), 0.0)

    def test_filter_by_score(self):
        docs = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.1}]
        kept = rerank_filter.filter_by_score(docs, threshold=0.3)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["text"], "a")

    def test_deduplicate(self):
        docs = [
            {"text": "劳动合同如何解除", "embedding": self.vectors[0]},
            {"text": "劳动合同如何解除的说明", "embedding": self.vectors[1]},
            {"text": "今天天气真好", "embedding": self.vectors[2]},
        ]
        unique = rerank_filter.deduplicate(docs, threshold=0.9)
        self.assertEqual(len(unique), 2)
        self.assertIn("天气", unique[1]["text"])

    def test_normalize_scores(self):
        normalized = rerank_filter.normalize_scores([{"score": 10.0}, {"score": 20.0}])
        self.assertEqual(normalized[0]["norm_score"], 0.0)
        self.assertEqual(normalized[1]["norm_score"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
