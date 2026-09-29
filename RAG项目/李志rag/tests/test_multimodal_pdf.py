import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app import rag_main


class MultimodalPdfTests(unittest.TestCase):
    def test_text_table_and_ocr_image_are_extractable(self):
        path = Path("output/pdf/multimodal_medical_test.pdf")
        if not path.exists():
            self.skipTest("Run scripts/generate_multimodal_pdf.py first")
        with patch.object(rag_main, "_safe_vl", return_value="视觉模型识别结果"):
            text = rag_main.extract_pdf_text(path)
        self.assertIn("血压由收缩压和舒张压组成", text)
        self.assertIn("测量前 | 安静休息 5 分钟", text)
        self.assertIn("OCR图片血压记录", text)
        self.assertIn("视觉模型识别结果", text)

    def test_vl_failure_keeps_basic_parser_available(self):
        with patch.object(rag_main, "_vl_image", side_effect=RuntimeError("test failure")):
            self.assertEqual(rag_main._safe_vl(Path("missing.png"), "ocr"), "")

    def test_chart_keywords_select_chart_task(self):
        self.assertEqual(rag_main._visual_task(Path("missing.png"), "各领域占比 30% 和 70%"), "chart")

    def test_vl_control_tokens_are_cleaned(self):
        result = rag_main._clean_vl_output("<fcel>项目<ucel>建议<nl><fcel>测量前<ecel>")
        self.assertNotIn("<fcel>", result)
        self.assertIn("测量前", result)

    def test_ocr_can_run_when_vl_is_disabled(self):
        path = Path("output/pdf/medical_knowledge_sample.pdf")
        if not path.exists():
            self.skipTest("Sample PDF is missing")
        with patch.object(rag_main.settings, "paddleocr_vl_enabled", False):
            text = rag_main.extract_pdf_text(path)
        self.assertTrue(text)

    def test_fast_extraction_skips_vl(self):
        path = Path("output/pdf/multimodal_medical_test.pdf")
        with patch.object(rag_main, "_safe_vl") as vl_mock:
            text = rag_main.extract_pdf_text(path, use_visual=False)
        self.assertIn("血压由收缩压和舒张压组成", text)
        vl_mock.assert_not_called()

    def test_native_table_does_not_repeat_visual_recognition(self):
        table = Mock()
        table.extract.return_value = [["项目", "建议"], ["测量前", "休息五分钟"]]
        with patch.object(rag_main, "_safe_vl") as vl_mock:
            text = rag_main._table_text(Mock(), Path("."), [table], use_visual=True)
        self.assertIn("测量前 | 休息五分钟", text)
        vl_mock.assert_not_called()

    def test_repeated_text_watermark_is_removed(self):
        pages = [
            "内部资料 仅供测试\n第一页正常正文",
            "内部资料 仅供测试\n第二页正常正文",
            "内部资料 仅供测试\n第三页正常正文",
        ]
        cleaned, removed = rag_main.remove_repeated_watermarks(pages)
        self.assertIn("内部资料 仅供测试", removed)
        self.assertNotIn("内部资料", "\n".join(cleaned))
        self.assertIn("第二页正常正文", cleaned[1])

    def test_table_rows_are_not_treated_as_watermarks(self):
        pages = ["项目 | 建议\n正文一", "项目 | 建议\n正文二", "项目 | 建议\n正文三"]
        cleaned, removed = rag_main.remove_repeated_watermarks(pages)
        self.assertNotIn("项目 | 建议", removed)
        self.assertIn("项目 | 建议", cleaned[0])


if __name__ == "__main__":
    unittest.main()
