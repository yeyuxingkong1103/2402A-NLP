# -*- coding: utf-8 -*-
"""DocumentQualityAssessmentSkill · 单元测试（不依赖外部数据）"""
import os
import shutil
import tempfile
import unittest

import pymupdf

from document_quality_assessment import Config, assess
from document_quality_assessment import detectors as D


def make_text_pdf(path, text="这是一段用于测试的中文文本。" * 40):
    doc = pymupdf.open()
    for _ in range(3):
        p = doc.new_page()
        p.insert_text((72, 72), text[:500], fontname="china-s")
    doc.save(path)
    doc.close()


def make_scan_pdf(path):
    """图片型：插入一张纯图像页（无文本层）→ 应判为扫描型。"""
    doc = pymupdf.open()
    p = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200))
    pix.set_rect(pix.irect, (200, 200, 200))
    p.insert_image(pymupdf.Rect(50, 50, 250, 250), pixmap=pix)
    doc.save(path)
    doc.close()


class SkillTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="dq_test_")
        make_text_pdf(os.path.join(self.d, "text.pdf"))
        make_scan_pdf(os.path.join(self.d, "scan.pdf"))
        # md5 完全重复
        shutil.copyfile(os.path.join(self.d, "text.pdf"), os.path.join(self.d, "text_copy.pdf"))
        # 敏感信息
        with open(os.path.join(self.d, "contact.md"), "w", encoding="utf-8") as f:
            f.write("联系人手机 13812345678，邮箱 test@example.com，身份证 11010119900307" + "1234" + "。")
        # 近重复
        with open(os.path.join(self.d, "v1.md"), "w", encoding="utf-8") as f:
            f.write("本专利涉及一种静电除尘器的外壳结构与管状入口设计。" * 10)
        with open(os.path.join(self.d, "v2.md"), "w", encoding="utf-8") as f:
            f.write("本专利涉及一种静电除尘器的外壳结构与管状入口设计。" * 10 + " 版本二。")
        self.cfg = Config()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_format_stats(self):
        rep = assess(self.d, self.cfg)
        ext = rep["format_stats"]["by_ext"]
        self.assertEqual(rep["format_stats"]["total_files"], 6)
        self.assertEqual(ext[".pdf"]["count"], 3)
        self.assertEqual(ext[".md"]["count"], 3)

    def test_pdf_types(self):
        rep = assess(self.d, self.cfg)
        pdf = rep["pdf_types"]
        self.assertGreaterEqual(pdf["scan"], 1)   # scan.pdf
        self.assertGreaterEqual(pdf["text"], 1)   # text.pdf / copy
        # 空页/纯图页不应有文本
        self.assertEqual(D.pdf_page_type(os.path.join(self.d, "scan.pdf"), self.cfg)["type"], "扫描型")

    def test_length_distribution(self):
        rep = assess(self.d, self.cfg)
        ld = rep["length_distribution"]
        self.assertEqual(ld["count"], 6)
        for k in ("P25", "P50", "P75", "P90", "P99"):
            self.assertIn(k, ld)

    def test_md5_duplicates(self):
        rep = assess(self.d, self.cfg)
        md5 = rep["duplicates"]["md5"]
        self.assertTrue(any(g["count"] == 2 for g in md5))

    def test_simhash_near(self):
        rep = assess(self.d, self.cfg)
        near = rep["duplicates"]["near"]
        self.assertTrue(len(near) >= 1)  # v1.md vs v2.md

    def test_sensitive(self):
        rep = assess(self.d, self.cfg)
        c = rep["sensitive"]["counts"]
        self.assertGreaterEqual(c.get("phone", 0), 1)
        self.assertGreaterEqual(c.get("email", 0), 1)
        self.assertGreaterEqual(c.get("idcard", 0), 1)
        # 银行卡默认关闭 → 不检测
        self.assertFalse(rep["sensitive"]["enabled"]["bankcard"])
        for it in rep["sensitive"]["items"]:
            self.assertIn("context", it)  # 每条附带上下文

    def test_classification(self):
        rep = assess(self.d, self.cfg)
        labels = rep["classification"]["label_counts"]
        self.assertTrue(any("Scan_PDF" in k for k in labels))


if __name__ == "__main__":
    unittest.main(verbosity=2)
