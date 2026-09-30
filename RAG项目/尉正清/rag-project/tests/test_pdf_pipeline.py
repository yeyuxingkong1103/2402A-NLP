# tests/test_pdf_pipeline.py
"""PDF 解析链路的单元测试：去水印 / 表格 / OCR 合并 / 编排。

夹具体现在 `tests/pdf_fixtures.py`，全部**现场合成**，不依赖
`data/sources/` 下的真实文档，也不写知识库，因此可以放进日常回归。

`PDFService.extract_pages()` 的用例一律把 MinerU 关掉，走兜底路径：
MinerU 是独立 venv 里的 7GB 重依赖，让单元测试依赖它既慢又不稳；
主路径的验证见 `docs/10-测试报告.md` §7.2（人工 + 端到端）。
"""
import pymupdf
import pytest

from app.core import mineru_service
from app.core.ocr_service import merge_ocr_text, _normalize
from app.core.pdf_service import PDFService
from app.core.table_service import get_table_service
from app.core.watermark import WatermarkRemover, WatermarkReport
from tests.pdf_fixtures import (BODY_MARK, PAGES, WM_TEXT, blank_doc,
                                noise_pixmap, save, text_of,
                                with_image_watermark, with_text_watermark)


# ---------------- 去水印 ----------------
class TestWatermarkText:
    """文字水印：多页同位置重复的浅灰短文本。"""

    def test_repeated_text_removed_without_touching_body(self, tmp_path):
        src = save(with_text_watermark(blank_doc()), tmp_path / "wm.pdf")
        before = text_of(src)
        assert before.count(WM_TEXT) == PAGES

        doc = pymupdf.open(src)
        report = WatermarkReport()
        WatermarkRemover().remove(doc, report)
        out = save(doc, tmp_path / "clean.pdf")

        after = text_of(out)
        assert after.count(WM_TEXT) == 0
        assert report.text_removed == PAGES
        # 正文不能被误删 —— §6.3 记过一次「正文保留率仅 5%」的回归
        assert BODY_MARK in after
        assert len(after) > len(before) * 0.8

    def test_repeated_short_wrapped_line_kept(self, tmp_path):
        """折行的续行很短，又会在多页重复出现——不能被当成水印删掉。

        实测缺陷：正文一句 11 字的折行续行（黑色、12 号、水平）在 12 页上
        重复出现于同一位置，被 `SHORT_TEXT` 规则判定为「具备水印形态」而删除，
        正文保留率掉到 79%。真水印是独立文本对象，折行续行与上一行同属一个
        多行块——`_watermark_like` 现在要求「极短」必须同时 `standalone`。
        """
        src = save(blank_doc(), tmp_path / "wrapped.pdf")

        doc = pymupdf.open(src)
        report = WatermarkReport()
        WatermarkRemover().remove(doc, report)
        out = save(doc, tmp_path / "clean.pdf")

        assert report.text_removed == 0
        assert BODY_MARK in text_of(out)

    def test_short_document_skips_detection(self, tmp_path):
        """不到 3 页不做水印统计——样本太少，重复不足以判定。"""
        src = save(with_text_watermark(blank_doc(pages=2)), tmp_path / "tiny.pdf")

        doc = pymupdf.open(src)
        report = WatermarkReport()
        WatermarkRemover().remove(doc, report)
        doc.close()

        assert report.text_removed == 0
        assert any("页数过少" in n for n in report.notes)


class TestWatermarkImage:
    """图片水印：多页重复铺版的 logo。"""

    def test_repeated_logo_removed(self, tmp_path):
        src = save(with_image_watermark(blank_doc()), tmp_path / "logo.pdf")
        before = text_of(src)

        doc = pymupdf.open(src)
        report = WatermarkReport()
        WatermarkRemover().remove(doc, report)
        out = save(doc, tmp_path / "clean.pdf")

        assert report.image_removed == PAGES
        assert BODY_MARK in text_of(out)
        assert len(text_of(out)) > len(before) * 0.8

    def test_tiny_solid_logo_ignored(self, tmp_path):
        """纯色小图压缩后不足 200 字节，指纹不可靠，宁可不删。"""
        doc = blank_doc()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
        pix.set_rect(pix.irect, (200, 30, 30))
        for page in doc:
            page.insert_image(pymupdf.Rect(20, 20, 60, 60), pixmap=pix)
        src = save(doc, tmp_path / "solid.pdf")

        doc = pymupdf.open(src)
        report = WatermarkReport()
        WatermarkRemover().remove(doc, report)
        doc.close()

        assert report.image_removed == 0


# ---------------- OCR 合并 ----------------
class TestMergeOcrText:
    """OCR 结果并入文本层：去重、噪声过滤、不留前导空行。"""

    def test_duplicate_line_not_repeated(self):
        """OCR 常把文本层已有句子再识别一遍，只是标点不同。"""
        merged, added = merge_ocr_text("本报告由世界卫生组织发布。",
                                       ["本报告由世界卫生组织发布"])
        assert added == 0
        assert merged == "本报告由世界卫生组织发布。"

    def test_novel_lines_appended(self):
        merged, added = merge_ocr_text("图 1 所示。",
                                       ["抑郁障碍患病率 3.8%", "焦虑障碍患病率 4.1%"])
        assert added == 2
        assert "【图表文字】" in merged
        assert "抑郁障碍患病率 3.8%" in merged

    def test_single_char_noise_dropped(self):
        _, added = merge_ocr_text("现有正文。", ["现有正文", "的", "了"])
        assert added == 0

    def test_empty_ocr_returns_original(self):
        assert merge_ocr_text("现有正文。", []) == ("现有正文。", 0)

    def test_empty_text_layer_has_no_leading_blank(self):
        """整页扫描件：文本层为空，正文不该以空行开头。"""
        merged, added = merge_ocr_text("", ["扫描页第一行", "第二行"])
        assert added == 2
        assert not merged.startswith("\n")
        assert merged.startswith("【图表文字】")

    def test_normalize_strips_punctuation(self):
        assert _normalize("抑郁障碍，患病率 3.8%！") == "抑郁障碍患病率38%"


# ---------------- 是否需要 OCR ----------------
class TestNeedsOcr:
    """`_needs_ocr` 决定这一页值不值得跑 OCR（CPU 上 10~40 秒/页）。"""

    @staticmethod
    def _page_with_image(box, with_text):
        doc = pymupdf.open()
        page = doc.new_page()
        if with_text:
            page.insert_textbox(pymupdf.Rect(60, 400, 540, 700),
                                "这一页有完整的正文段落。" * 6,
                                fontname="china-s", fontsize=12)
        page.insert_image(box, pixmap=noise_pixmap(80))
        return doc, page

    def test_scanned_page_needs_ocr(self):
        """文本层几乎为空 + 整页是图 → 扫描页。"""
        doc, page = self._page_with_image(pymupdf.Rect(0, 0, 595, 842), False)
        assert PDFService._needs_ocr(page, 0) is True
        doc.close()

    def test_chart_page_needs_ocr(self):
        """图片占 15%~95% → 图表页。"""
        doc, page = self._page_with_image(pymupdf.Rect(60, 60, 400, 400), True)
        assert PDFService._needs_ocr(page, 300) is True
        doc.close()

    def test_full_page_background_with_text_excluded(self):
        """铺满整页的背景图不算图表页——跑出来全是噪声。"""
        doc, page = self._page_with_image(pymupdf.Rect(0, 0, 595, 842), True)
        assert PDFService._needs_ocr(page, 300) is False
        doc.close()

    def test_page_without_image_skipped(self):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(60, 60, 540, 700), "纯文字页。",
                            fontname="china-s", fontsize=12)
        assert PDFService._needs_ocr(page, 10) is False
        doc.close()


# ---------------- 表格 ----------------
class TestTableService:

    @staticmethod
    def _table_pdf(path):
        doc = pymupdf.open()
        page = doc.new_page()
        data = [["产品", "2023", "2024", "增幅"],
                ["稳健型", "3.2%", "3.8%", "+0.6"],
                ["进取型", "5.1%", "4.4%", "-0.7"]]
        for r, row in enumerate(data):
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(60 + c * 120, 100 + r * 30,
                                    60 + (c + 1) * 120, 100 + (r + 1) * 30)
                page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                page.insert_textbox(rect, cell, fontname="china-s",
                                    fontsize=10, align=1)
        return save(doc, path)

    def test_multicolumn_table_extracted_as_markdown(self, tmp_path):
        src = self._table_pdf(tmp_path / "t.pdf")
        got = get_table_service().extract_document_tables(src, 1)
        assert 1 in got and len(got[1]) == 1
        md = got[1][0]
        assert md.count("|") > 12          # 4 列 × 3 行
        assert "稳健型" in md and "3.8%" in md

    def test_single_column_block_rejected(self, tmp_path):
        """单列表格几乎都是带框标题——收进来只会在【表格数据】里制造噪声。"""
        doc = pymupdf.open()
        page = doc.new_page()
        for r in range(3):
            rect = pymupdf.Rect(60, 100 + r * 30, 420, 130 + r * 30)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_textbox(rect, "封面标题第 %d 行" % (r + 1),
                                fontname="china-s", fontsize=10)
        src = save(doc, tmp_path / "one.pdf")

        assert get_table_service().extract_document_tables(src, 1) == {}

    def test_merge_into_text_appends_section(self):
        merged, n = get_table_service().merge_into_text("正文段落。", ["| a | b |"])
        assert n == 1
        assert merged.startswith("正文段落。")
        assert "【表格数据】" in merged

    def test_merge_no_tables_is_noop(self):
        assert get_table_service().merge_into_text("正文。", []) == ("正文。", 0)

    def test_unavailable_service_degrades(self, monkeypatch):
        """pdfplumber 不可用时返回空，不能抛错打断整条兜底路径。"""
        svc = get_table_service()
        monkeypatch.setattr(svc, "available", lambda: False)
        assert svc.extract_document_tables("whatever.pdf", 3) == {}


# ---------------- 编排 ----------------
class TestExtractPagesFallback:
    """`extract_pages()` 兜底路径的端到端编排（MinerU 关闭）。"""

    @pytest.fixture(autouse=True)
    def _no_mineru(self, monkeypatch):
        monkeypatch.setattr(mineru_service, "available", lambda: False)

    def test_watermark_removed_from_output(self, tmp_path):
        """兜底路径必须读**去过水印的副本**。

        实测缺陷：兜底路径读的是调用方传进来的原文件，水印擦在了另一份副本上，
        于是报告 `text_removed` 照常计数、正文里水印却原样还在。
        """
        src = save(with_text_watermark(blank_doc()), tmp_path / "wm.pdf")

        pages, report = PDFService().extract_pages(src, use_ocr=False)
        joined = "\n".join(p["text"] for p in pages)

        assert report.path == "pymupdf"
        assert report.text_removed == PAGES
        assert WM_TEXT not in joined
        assert BODY_MARK in joined
        assert len(pages) == PAGES

    def test_no_watermark_keeps_all_text(self, tmp_path):
        src = save(blank_doc(), tmp_path / "plain.pdf")
        pages, report = PDFService().extract_pages(src, use_ocr=False)
        assert report.text_removed == 0
        assert BODY_MARK in "\n".join(p["text"] for p in pages)
        assert len(pages) == PAGES

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PDFService().extract_pages("/tmp/definitely_not_here.pdf")
