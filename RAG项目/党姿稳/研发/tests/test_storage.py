"""
test_storage.py — 单元测试（存储与解析部分）

覆盖向量库读写、知识库导入检索、PDF 解析。同样离线运行，
不调用大模型、不依赖 Redis / Milvus：

    python -m unittest tests.test_storage -v
    python tests/test_storage.py

没有测试 PDF 时，解析用例会自动跳过（先跑 scripts/generate_test_pdfs.py 生成）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  （必须先执行，才能安全导入业务模块）

import config  # noqa: E402
import embeddings  # noqa: E402
import quality  # noqa: E402
from _bootstrap import TEMP_DIR  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402
from local_store import LocalStore  # noqa: E402
from milvus_store import MilvusStore  # noqa: E402


class TestVectorStore(unittest.TestCase):
    """本地向量库的写入、检索、删除。"""

    COLLECTION = "_unit_test"

    def setUp(self):
        self.store = LocalStore(TEMP_DIR / "test_store.json")
        self.store.drop(self.COLLECTION)

    def _upsert(self, records: list[dict]) -> list[int]:
        return self.store.upsert(self.COLLECTION, records)

    def _record(self, text: str, **extra) -> dict:
        return {"text": text, "embedding": embeddings.encode_query(text), **extra}

    def test_upsert_and_search(self):
        ids = self._upsert([self._record("劳动合同解除需要支付经济补偿", source="a.pdf"),
                            self._record("今天天气很好适合散步", source="a.pdf")])
        self.assertEqual(len(ids), 2)
        self.assertEqual(self.store.count(self.COLLECTION), 2)

        hits = self.store.search(self.COLLECTION, embeddings.encode_query("劳动合同赔偿"), top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("劳动", hits[0]["text"])

    def test_filter_by_field(self):
        self._upsert([self._record("用户甲的偏好", user_id="u1"),
                      self._record("用户乙的偏好", user_id="u2")])
        hits = self.store.search(
            self.COLLECTION, embeddings.encode_query("偏好"), top_k=5, filters={"user_id": "u1"}
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["user_id"], "u1")

    def test_delete_and_count(self):
        self._upsert([self._record("待删除", user_id="u9")])
        self.assertEqual(self.store.delete(self.COLLECTION, {"user_id": "u9"}), 1)
        self.assertEqual(self.store.count(self.COLLECTION), 0)

    def test_upsert_is_idempotent(self):
        record = self._record("重复内容", source="same.pdf")
        self._upsert([record])
        self._upsert([record])
        self.assertEqual(self.store.count(self.COLLECTION), 1)

    def test_page_roundtrip(self):
        """page 要能原样存取，回答末尾的"第N页"引用靠它。"""
        self._upsert([self._record("第三页的法条内容", source="a.pdf", page=3)])
        rows = self.store.query(self.COLLECTION, limit=10)
        self.assertEqual(rows[0]["page"], 3)


class _NormalizeStub:
    """借用 MilvusStore 的字段裁剪逻辑，不建立任何真实连接。"""

    _fields = MilvusStore._fields
    _normalize = MilvusStore._normalize


class TestMilvusNormalize(unittest.TestCase):
    """Milvus 的字段裁剪。

    集合关掉了动态字段，schema 外的字段会被静默丢掉——
    也就是说字段表漏一个，数据就无声无息地没了，必须逐字段验。
    """

    def test_kb_page_kept(self):
        row = MilvusStore._normalize(
            _NormalizeStub(),
            "kb_legal",
            {"text": "劳动法条文", "source": "a.pdf", "page": 7, "embedding": [0.1]},
        )
        self.assertEqual(row["page"], 7)
        self.assertEqual(row["source"], "a.pdf")

    def test_kb_page_defaults_to_zero(self):
        row = MilvusStore._normalize(
            _NormalizeStub(), "kb_legal", {"text": "无页码的纯文本", "embedding": [0.1]}
        )
        self.assertEqual(row["page"], 0)

    def test_memory_domain_kept(self):
        row = MilvusStore._normalize(
            _NormalizeStub(),
            config.LONG_TERM_COLLECTION,
            {
                "text": "用户偏好",
                "user_id": "u_1",
                "domain": "legal",
                "fact_type": "profile",
                "embedding": [0.1],
            },
        )
        self.assertEqual(row["domain"], "legal")
        self.assertEqual(row["fact_type"], "profile")

    def test_unknown_field_dropped(self):
        row = MilvusStore._normalize(
            _NormalizeStub(),
            "kb_legal",
            {"text": "x", "embedding": [0.1], "schema里没有的字段": 1},
        )
        self.assertNotIn("schema里没有的字段", row)


class TestKnowledgeBase(unittest.TestCase):
    """质量过滤与导入检索。"""

    def setUp(self):
        self.kb = KnowledgeBase()
        self.kb.clear("legal")

    def test_quality_filter(self):
        chunks = [
            {"text": "这是一段足够长的正常法律条文内容。"},
            {"text": "太短"},  # 长度不足
            {"text": "。。。。。。。。。。。。"},  # 无意义
            {"text": "䷰䷱䷲䷳䷴䷵䷶䷷䷸䷹䷺䷻䷼䷽䷾"},  # 乱码
        ]
        kept = quality.filter_chunks(chunks)
        self.assertEqual(len(kept), 1)
        self.assertIn("正常法律条文", kept[0]["text"])

    def test_import_pdf_keeps_page(self):
        """页码必须是独立字段，source 只留文件名，才拼得出引用。"""
        pdfs = sorted(config.GENERATED_PDF_DIR.glob("*.pdf"))
        if not pdfs:
            self.skipTest("未找到测试 PDF，先运行 python scripts/generate_test_pdfs.py")

        report = self.kb.import_pdf(pdfs[0], "legal")
        self.assertGreater(report["stored"], 0)

        from vector_store import get_store

        rows = get_store().query("kb_legal", limit=10000)
        pages = {int(row.get("page") or 0) for row in rows}
        self.assertTrue(pages - {0}, "页码没有写进向量库")
        self.assertEqual(
            {row.get("source") for row in rows}, {pdfs[0].name}, "source 应该只是文件名"
        )

    def test_import_and_search(self):
        stored = self.kb.import_texts(
            [
                "劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。",
                "用人单位应当在解除或者终止劳动合同时出具解除或者终止劳动合同的证明。",
            ],
            domain="legal",
            source="unit_test.txt",
        )
        self.assertGreater(stored, 0)

        results = self.kb.search("怎么辞职", "legal", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertTrue(any("解除劳动合同" in r["text"] for r in results))

    def test_unknown_domain_returns_empty(self):
        self.assertEqual(self.kb.search("任意问题", "not-a-domain"), [])

    def test_clear(self):
        self.kb.import_texts(["一段用于测试清除的知识库文本内容。"], domain="legal", source="t.txt")
        self.kb.clear("legal")
        self.assertEqual(self.kb.stats()["legal"]["chunks"], 0)


class TestPDFParser(unittest.TestCase):
    """PDF 解析。没有测试文件时自动跳过。"""

    def _parser(self, **kwargs):
        from pdf_parser import PDFParser

        return PDFParser(use_mineru=False, use_ocr=False, **kwargs)

    def test_parse_generated_pdf(self):
        pdfs = sorted(config.GENERATED_PDF_DIR.glob("*.pdf"))
        if not pdfs:
            self.skipTest("未找到测试 PDF，先运行 python scripts/generate_test_pdfs.py")

        blocks = self._parser().parse(pdfs[0])
        self.assertGreater(len(blocks), 0, "解析结果为空")
        self.assertTrue(all(b["page"] >= 1 for b in blocks))
        self.assertTrue(all(b["type"] in {"text", "table", "image"} for b in blocks))
        self.assertGreater(len("".join(b["text"] for b in blocks)), 500)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self._parser().parse(TEMP_DIR / "not-exist.pdf")

    def test_non_pdf_rejected(self):
        target = TEMP_DIR / "fake.txt"
        target.write_text("hello", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._parser().parse(target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
