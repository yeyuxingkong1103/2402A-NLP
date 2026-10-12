# -*- coding: utf-8 -*-
"""DocumentQualityAssessmentSkill 单元测试"""
import os
import sys
import tempfile
import unittest

# 把 skill 根目录加入 path
sys.path.insert(0, "/ragflow/document_quality_assessment")

from assess import (
    format_stats,
    md5_file,
    simhash,
    hamming,
    scan_sensitive,
    classify_docs,
    classify_pdf,
    get_length,
)


class TestFormatStats(unittest.TestCase):
    def test_empty(self):
        r = format_stats([])
        self.assertEqual(r["total_files"], 0)

    def test_distribution(self):
        files = ["a.pdf", "b.pdf", "c.md", "d.docx"]
        r = format_stats(files)
        self.assertEqual(r["total_files"], 4)
        d = {x["ext"]: x["count"] for x in r["distribution"]}
        self.assertEqual(d["pdf"], 2)
        self.assertEqual(d["md"], 1)


class TestMD5(unittest.TestCase):
    def test_identical_files(self):
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"hello world")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"hello world")
            p2 = f2.name
        self.assertEqual(md5_file(p1), md5_file(p2))
        os.unlink(p1); os.unlink(p2)

    def test_different_files(self):
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"aaaa")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"bbbb")
            p2 = f2.name
        self.assertNotEqual(md5_file(p1), md5_file(p2))
        os.unlink(p1); os.unlink(p2)


class TestSimHash(unittest.TestCase):
    def test_identical(self):
        h1 = simhash("这是一个测试文档")
        h2 = simhash("这是一个测试文档")
        self.assertEqual(hamming(h1, h2), 0)

    def test_similar(self):
        h1 = simhash("这是一个测试文档的内容用于验证相似度")
        h2 = simhash("这是一个测试文档的内容用于验证相似程度")
        self.assertLess(hamming(h1, h2), 30)

    def test_empty(self):
        self.assertEqual(simhash(""), 0)


class TestSensitiveScan(unittest.TestCase):
    def test_phone_and_email(self):
        cfg = {
            "phone": {"enabled": True, "pattern": "1[3-9]\\d{9}"},
            "email": {"enabled": True, "pattern": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"},
            "id_card": {"enabled": False, "pattern": ""},
            "bank_card": {"enabled": False, "pattern": ""},
            "context_chars": 20,
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("电话 13812345678，邮箱 test@example.com")
            path = f.name
        r = scan_sensitive([path], cfg)
        self.assertEqual(r["summary"]["phone_hits"], 1)
        self.assertEqual(r["summary"]["email_hits"], 1)
        # 上下文可操作性
        item = r["pending_items"][0]
        self.assertIn("file", item)
        self.assertIn("context", item)
        os.unlink(path)

    def test_bank_card_disabled(self):
        cfg = {
            "phone": {"enabled": False, "pattern": ""},
            "email": {"enabled": False, "pattern": ""},
            "id_card": {"enabled": False, "pattern": ""},
            "bank_card": {"enabled": False, "pattern": "\\d{16,19}"},
            "context_chars": 20,
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write("卡号 6222021234567890123")
            path = f.name
        r = scan_sensitive([path], cfg)
        self.assertEqual(r["summary"]["total_hits"], 0)
        os.unlink(path)


class TestClassifier(unittest.TestCase):
    def test_scan_pdf_label(self):
        files = ["/tmp/fake.pdf"]
        pdf_info = {"details": [{"path": "/tmp/fake.pdf", "type": "scan", "has_error": False}]}
        dedup_result = {"exact_duplicates": []}
        r = classify_docs(files, pdf_info, dedup_result, {})
        self.assertEqual(r["documents"][0]["primary_label"], "Scan_PDF")

    def test_parse_failed_label(self):
        files = ["/tmp/broken.pdf"]
        pdf_info = {"details": [{"path": "/tmp/broken.pdf", "has_error": True}]}
        dedup_result = {"exact_duplicates": []}
        r = classify_docs(files, pdf_info, dedup_result, {})
        self.assertEqual(r["documents"][0]["primary_label"], "Parse_Failed")

    def test_duplicate_label(self):
        files = ["/tmp/a.pdf"]
        pdf_info = {"details": [{"path": "/tmp/a.pdf", "type": "text", "has_error": False}]}
        dedup_result = {"exact_duplicates": [{"md5": "x", "files": ["/tmp/a.pdf"]}]}
        r = classify_docs(files, pdf_info, dedup_result, {})
        self.assertIn("Duplicate", r["documents"][0]["labels"])


class TestPDFClassifier(unittest.TestCase):
    def test_missing_file(self):
        cfg = {"max_pages_per_file": 10, "scan_page_char_threshold": 50}
        r = classify_pdf("/tmp/nonexistent_file.pdf", cfg)
        self.assertTrue(r["has_error"])
        self.assertEqual(r["type"], "parse_failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)