import json
import tempfile
import unittest
from pathlib import Path


class CorpusTests(unittest.TestCase):
    def test_report_identity_uses_stock_code_not_mojibake_name(self):
        from corpus import report_for_path

        report = report_for_path(Path("乱码__600036__乱码.txt"))

        self.assertEqual(report.stock_code, "600036")
        self.assertEqual(report.company, "招商银行")
        self.assertEqual(report.year, 2019)

    def test_iter_jsonl_rows_skips_blank_and_malformed_lines(self):
        from corpus import iter_jsonl_rows

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"page": 3, "type": "text", "inside": "有效文本"}, ensure_ascii=False),
                        "{broken-json",
                        json.dumps({"page": 4, "type": "text", "inside": ""}, ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )

            rows, skipped = iter_jsonl_rows(path)

        self.assertEqual(rows, [{"page": 3, "type": "text", "inside": "有效文本"}])
        self.assertEqual(skipped, 2)

    def test_load_corpus_keeps_document_and_page_metadata(self):
        from corpus import load_corpus

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_dir = root / "txt"
            text_dir.mkdir()
            source = text_dir / "any__000001__name.txt"
            source.write_text(
                "\n".join(
                    json.dumps(
                        {"page": page, "type": "text", "inside": text},
                        ensure_ascii=False,
                    )
                    for page, text in [(7, "董事长致辞"), (7, "盈利增长来自零售转型。"), (8, "风险管理持续加强。")]
                ),
                encoding="utf-8",
            )

            chunks = load_corpus(root, max_chars=40, overlap=0)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].stock_code, "000001")
        self.assertEqual(chunks[0].document, "平安银行2019年年报")
        self.assertEqual(chunks[0].page, 7)
        self.assertIn("盈利增长", "".join(chunk.text for chunk in chunks))

    def test_load_corpus_reports_missing_txt_directory(self):
        from corpus import load_corpus

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "txt"):
                load_corpus(Path(tmp))


if __name__ == "__main__":
    unittest.main()
