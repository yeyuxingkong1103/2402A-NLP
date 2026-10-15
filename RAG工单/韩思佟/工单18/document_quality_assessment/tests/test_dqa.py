# -*- coding: utf-8 -*-
"""
工单18 —— DocumentQualityAssessmentSkill 单元测试

覆盖六大功能 + 配置化 + 错误处理：
  1. 格式分布统计
  2. PDF 页面类型识别（文字型/扫描型/混合型 + 待确认列表）
  3. 文档长度分布（分位数 + 区间）
  4. 重复检测（MD5 精确 + SimHash 近似）
  5. 敏感信息检测（待审核列表 + 银行卡默认关闭）
  6. 文档分类标签（含 OCR 路由）
  7. 阈值配置化（不硬编码）
  8. 损坏 PDF 的容错

运行：
  python tests/test_dqa.py
  # 或
  python -m pytest tests/test_dqa.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from dqa import (analyze_format_distribution, analyze_length_distribution,   # noqa: E402
                 classify_pdf_pages, detect_duplicates,
                 detect_sensitive_info, classify_documents,
                 build_report, to_html, to_markdown)
from dqa.dedup import md5_of, simhash64, hamming                            # noqa: E402
from dqa.sensitive import PATTERNS, ENABLED_BY_DEFAULT                      # noqa: E402

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

LONG_TEXT = ("这是一段用于测试的正文中文字幕内容，需要足够长才能让页面字符数超过阈值，"
             "从而被判定为文字型PDF，而不是扫描型PDF，这样测试断言才有意义。")


def make_pdf(path: Path, pages: int = 2, text: str | None = LONG_TEXT):
    """造一个 PDF；text=None 表示造扫描件（只有图、没有文本层）"""
    if fitz is None:
        raise unittest.SkipTest("PyMuPDF 未安装，跳过涉及构造 PDF 的用例")
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            # 中文字体要用内置 'china-s'，且分页宽换行写入，保证字符数足够
            body = text * 3
            y = 100
            for i in range(0, len(body), 30):
                page.insert_text((72, y), body[i:i + 30],
                                 fontname="china-s", fontsize=11)
                y += 24
        else:
            # 画一个矩形当"图"，不写任何文字 -> 扫描页
            page.draw_rect(fitz.Rect(72, 100, 300, 300), fill=(0.9, 0.9, 0.9))
    doc.save(str(path))
    doc.close()


class TestFormatStats(unittest.TestCase):
    def test_extension_and_group(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.pdf").write_bytes(b"x")
            (root / "b.docx").write_bytes(b"x")
            (root / "c.md").write_text("hello", encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub" / "d.pdf").write_bytes(b"x")
            r = analyze_format_distribution(root)
            self.assertEqual(r["文件总数"], 4)
            self.assertEqual(r["按扩展名"][".pdf"]["数量"], 2)
            self.assertEqual(r["按类别"]["PDF"]["数量"], 2)
            self.assertEqual(r["按类别"]["文档"]["数量"], 1)
            self.assertAlmostEqual(r["按类别"]["PDF"]["占比"], 0.5, places=2)

    def test_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            analyze_format_distribution(Path("/not/exist/xxx"))


class TestPdfType(unittest.TestCase):
    def test_text_pdf(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "t.pdf"
            make_pdf(p, pages=2, text="这是一段普通正文，用来判定文字型PDF。")
            r = classify_pdf_pages(p)
            self.assertEqual(r["type"], "text")
            self.assertEqual(r["scanned_pages"], [])
            self.assertEqual(r["pages"], 2)

    def test_scanned_pdf(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "s.pdf"
            make_pdf(p, pages=3, text=None)
            r = classify_pdf_pages(p)
            self.assertEqual(r["type"], "scanned")
            self.assertEqual(len(r["scanned_pages"]), 3)
            self.assertGreaterEqual(r["scanned_ratio"], 0.7)

    def test_mixed_pdf(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "m.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 120), "有文字的一页" * 30, fontsize=11)
            page = doc.new_page()
            page.draw_rect(fitz.Rect(72, 100, 300, 300), fill=(0.9, 0.9, 0.9))
            doc.save(str(p)); doc.close()
            r = classify_pdf_pages(p)
            self.assertEqual(r["type"], "mixed")

    def test_threshold_is_configurable(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "t.pdf"
            make_pdf(p, pages=1, text="中等长度的一段正文内容" * 5)   # 55 字
            # 阈值调大 -> 同一文件从 text 变 scanned，证明阈值没有硬编码
            self.assertEqual(classify_pdf_pages(p, min_chars_per_page=2)["type"], "text")
            self.assertEqual(classify_pdf_pages(p, min_chars_per_page=10000)["type"],
                             "scanned")

    def test_broken_pdf_not_crash(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "bad.pdf"
            p.write_bytes(b"not a pdf at all")
            r = classify_pdf_pages(p)
            self.assertTrue(r["needs_review"])
            self.assertIn("无法打开PDF", r["error"])


class TestLength(unittest.TestCase):
    def test_percentiles(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            for i, n in enumerate([100, 500, 1000, 5000, 20000, 100000]):
                (root / f"f{i}.txt").write_text("字" * n, encoding="utf-8")
            r = analyze_length_distribution(root, exts={".txt"})
            self.assertEqual(r["统计文件数"], 6)
            self.assertGreaterEqual(r["P50"], r["P25"] >= 0 * 1)
            self.assertGreaterEqual(r["P90"], r["P50"])
            self.assertGreaterEqual(r["P99"], r["P90"])
            self.assertEqual(r["最小值"], 100)
            self.assertEqual(r["最大值"], 100000)
            self.assertEqual(sum(b["数量"] for b in r["长度区间分布"]), 6)


class TestDedup(unittest.TestCase):
    def test_md5_exact(self):
        with TemporaryDirectory() as d:
            a, b = Path(d) / "a.pdf", Path(d) / "b.pdf"
            make_pdf(a, pages=1, text="同一份文件内容")
            b.write_bytes(a.read_bytes())
            self.assertEqual(md5_of(a), md5_of(b))

    def test_simhash_similarity(self):
        h1 = simhash64("这是一段测试文本，用于验证 SimHash 的相似度计算逻辑。")
        h2 = simhash64("这是一段测试文本，用于验证 SimHash 的相似度计算逻辑！")
        h3 = simhash64("完全不相关的一段话，内容差异很大，主题也完全不一样哦。")
        self.assertLessEqual(hamming(h1, h2), 12)
        self.assertGreater(hamming(h1, h3), hamming(h1, h2))

    def test_detect_duplicates_structure(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            a = root / "a.txt"
            a.write_text("同一份文档的第一版内容" * 20, encoding="utf-8")
            (root / "b.txt").write_text(a.read_text(encoding="utf-8"),
                                        encoding="utf-8")
            r = detect_duplicates(root, exts={".txt"})
            self.assertEqual(r["MD5精确重复组数"], 1)
            self.assertEqual(r["MD5可清理文件数"], 1)
            self.assertIn("待确认的版本冲突列表", r)


class TestSensitive(unittest.TestCase):
    def test_phone_and_email_detected(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "s.txt"
            p.write_text("联系人张三，手机13812345678，邮箱 zhangsan@example.com",
                         encoding="utf-8")
            r = detect_sensitive_info(Path(d), exts={".txt"})
            hits = r["待审核敏感信息列表"]
            self.assertTrue(any(h["类型"] == "手机号" for h in hits))
            self.assertTrue(any(h["类型"] == "邮箱" for h in hits))
            # 每条都必须带上下文，操作员才能判断
            for h in hits:
                self.assertIn("上下文", h)
                self.assertIn("文件", h)

    def test_bank_card_disabled_by_default(self):
        self.assertFalse(ENABLED_BY_DEFAULT["银行卡"])
        with TemporaryDirectory() as d:
            p = Path(d) / "s.txt"
            p.write_text("银行卡号6222021234567890123", encoding="utf-8")
            r = detect_sensitive_info(Path(d), exts={".txt"})
            self.assertEqual(r["各类型命中数"]["银行卡"], 0)
            # 手动开启后应能检出
            r2 = detect_sensitive_info(Path(d), exts={".txt"},
                                       enable={"手机号": True, "邮箱": True,
                                               "身份证": True, "银行卡": True})
            self.assertGreaterEqual(r2["各类型命中数"]["银行卡"], 1)


class TestClassifier(unittest.TestCase):
    def test_ocr_routing(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            p = root / "scan.pdf"
            make_pdf(p, pages=2, text=None)
            fmt = analyze_format_distribution(root)
            from dqa.pdf_type import scan_pdf_types
            pt = scan_pdf_types(root)
            ln = analyze_length_distribution(root, exts={".pdf"})
            lb = classify_documents(fmt["_files"], pt, ln)
            self.assertEqual(lb["需要OCR路由的文件数"], 1)
            first = lb["标签清单"][0]
            self.assertIn("Scan_PDF", first["tags"])
            self.assertIn("OCR", first["route"])


class TestReport(unittest.TestCase):
    def test_report_and_exports(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.txt").write_text("内容" * 50, encoding="utf-8")
            fmt = analyze_format_distribution(root)
            pt = {"类型分布": {}, "待确认列表": []}
            ln = analyze_length_distribution(root, exts={".txt"})
            dd = detect_duplicates(root, exts={".txt"})
            se = detect_sensitive_info(root, exts={".txt"})
            lb = classify_documents(fmt["_files"], pt, ln, dd, se)
            rep = build_report(root, fmt, pt, ln, dd, se, lb, 0.1)
            self.assertIn("总览", rep)
            for k in ("一_格式分布统计", "二_PDF页面类型识别", "三_文档长度分布",
                      "四_重复检测", "五_敏感信息检测", "六_文档分类标签"):
                self.assertIn(k, rep)
            html = to_html(rep)
            self.assertIn("<html", html)
            md = to_markdown(rep)
            self.assertIn("# 知识库文档质量评估报告", md)
            # 内部大字段不应出现在最终报告里
            self.assertNotIn("_files", rep["一_格式分布统计"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
