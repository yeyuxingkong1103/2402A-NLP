import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from corpus import load_zip_corpus, report_for_name


class ZipCorpusTests(unittest.TestCase):
    def _make_zip(self, path: Path):
        rows = [
            {"page": 7, "type": "text", "inside": "主要会计数据"},
            {"page": 7, "type": "text", "inside": "2019年营业收入为1,379.58亿元，净利润281.95亿元。"},
        ]
        payload = "\n".join([json.dumps(row, ensure_ascii=False) for row in rows] + ["{bad json", ""])
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("ccf_competition/txt/乱码__000001__2019年报.txt", payload.encode("utf-8"))

    def test_zip_loader_keeps_report_page_and_skips_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.zip"
            self._make_zip(path)
            chunks = load_zip_corpus(path, max_chars=200, overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].stock_code, "000001")
        self.assertEqual(chunks[0].company, "平安银行")
        self.assertEqual(chunks[0].year, 2019)
        self.assertEqual(chunks[0].page, 7)
        self.assertIn("1,379.58", chunks[0].text)

    def test_report_identity_uses_stock_code_even_when_filename_is_mojibake(self):
        report = report_for_name("2020__����__601318__����.txt")
        self.assertEqual((report.company, report.year), ("中国平安", 2019))

    def test_unknown_report_and_empty_zip_have_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "股票代码"):
            report_for_name("unknown.txt")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.zip"
            with zipfile.ZipFile(path, "w"):
                pass
            with self.assertRaisesRegex(ValueError, "没有可用"):
                load_zip_corpus(path)


if __name__ == "__main__":
    unittest.main()

