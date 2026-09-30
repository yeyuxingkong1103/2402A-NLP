"""四份演示 TXT 必须来自各自解析器，并清楚标出 PDF 节选。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import export_parser_samples as samples


class ParserSamplesTests(unittest.TestCase):
    def test_exports_four_named_results_without_touching_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "guide.pdf"
            source.write_bytes(b"%PDF-1.4")
            output = root / "samples"
            with patch.object(samples, "load_env"), \
                 patch.object(samples, "first_pages", return_value=3), \
                 patch.object(samples, "pdfplumber_text", return_value="表格文字"), \
                 patch.object(samples, "parse_document",
                              side_effect=[("普通文字", "pymupdf"),
                                           ("识别文字", "paddleocr"),
                                           ("版式文字", "mineru")]) as parser:
                written = samples.export_samples(source, output, 3)
            self.assertEqual(set(samples.PARSERS), set(written))
            self.assertEqual(3, parser.call_count)
            self.assertEqual(b"%PDF-1.4", source.read_bytes())
            for name, path in written.items():
                content = path.read_text(encoding="utf-8")
                self.assertIn(f"解析器：{name}", content)
                self.assertIn("第1-3页", content)
                self.assertIn("节选", content)

    def test_failed_parser_does_not_create_misleading_txt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "guide.pdf"
            source.write_bytes(b"%PDF-1.4")
            with patch.object(samples, "load_env"), \
                 patch.object(samples, "first_pages", return_value=1), \
                 patch.object(samples, "parse_document", side_effect=RuntimeError("依赖缺失")):
                written = samples.export_samples(source, root / "samples", 1,
                                                 ("paddleocr",))
            self.assertEqual({}, written)
            self.assertFalse((root / "samples/paddleocr.txt").exists())


if __name__ == "__main__":
    unittest.main()
