import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import embed_chunks
import parse_pdf


class ParsePdfValidationTests(unittest.TestCase):
    def test_rejects_pdf_larger_than_200mb(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "large.pdf"
            pdf_path.write_bytes(b"0")
            with patch.object(Path, "stat") as stat:
                stat.return_value.st_size = parse_pdf.MAX_PDF_BYTES + 1
                with self.assertRaisesRegex(parse_pdf.ParsePdfError, "超过 200MB"):
                    parse_pdf.validate_pdf(pdf_path)

    def test_rejects_pdf_larger_than_600_pages(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "too_many_pages.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            with patch.object(parse_pdf, "count_pdf_pages", return_value=parse_pdf.MAX_PDF_PAGES + 1):
                with self.assertRaisesRegex(parse_pdf.ParsePdfError, "超过 600 页"):
                    parse_pdf.validate_pdf(pdf_path)
    def test_parse_pdf_with_api_downloads_markdown_and_content_list(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "guide.pdf"
            output_dir = temp_path / "parsed"
            pdf_path.write_bytes(b"%PDF-1.4")

            responses = iter(
                [
                    {"batch_id": "batch-1"},
                    {
                        "status": "done",
                        "markdown_url": "https://example.test/guide.md",
                        "content_list_url": "https://example.test/content_list.json",
                    },
                ]
            )

            def fake_json_request(url, token, payload=None, method="POST"):
                return next(responses)

            def fake_download(url, token):
                if url.endswith("guide.md"):
                    return "# 指南\n正文".encode("utf-8")
                if url.endswith("content_list.json"):
                    return b"[]"
                raise AssertionError(url)

            with patch.dict(os.environ, {"MINERU_TOKEN": "token"}), patch.object(
                parse_pdf, "validate_pdf", return_value=(8, 1)
            ), patch.object(parse_pdf, "api_json_request", side_effect=fake_json_request), patch.object(
                parse_pdf, "download_api_file", side_effect=fake_download
            ):
                markdown_path, content_list_path = parse_pdf.parse_pdf(pdf_path, output_dir)

            self.assertEqual(markdown_path.read_text(encoding="utf-8"), "# 指南\n正文")
            self.assertEqual(json.loads(content_list_path.read_text(encoding="utf-8")), [])
    def test_parse_pdf_strips_trailing_whitespace_from_markdown_filename(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "guide .pdf"
            output_dir = temp_path / "parsed"
            pdf_path.write_bytes(b"%PDF-1.4")

            responses = iter(
                [
                    {"batch_id": "batch-1"},
                    {
                        "status": "done",
                        "markdown_url": "https://example.test/guide.md",
                        "content_list_url": "https://example.test/content_list.json",
                    },
                ]
            )

            with patch.dict(os.environ, {"MINERU_TOKEN": "token"}), patch.object(
                parse_pdf, "validate_pdf", return_value=(8, 1)
            ), patch.object(parse_pdf, "api_json_request", side_effect=lambda *args, **kwargs: next(responses)), patch.object(
                parse_pdf, "download_api_file", side_effect=lambda url, token: "# 指南".encode("utf-8") if url.endswith(".md") else b"[]"
            ):
                markdown_path, content_list_path = parse_pdf.parse_pdf(pdf_path, output_dir)

            self.assertEqual(markdown_path.name, "guide.md")
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), "# 指南")
            self.assertTrue(content_list_path.exists())

        class FakeResponse:
            status = 200

            def read(self):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        class FakeConnection:
            def __init__(self, host, timeout):
                self.host = host
                self.timeout = timeout
                self.requests = []

            def request(self, method, path, body=None, headers=None):
                self.requests.append((method, path, body, headers or {}))

            def getresponse(self):
                return FakeResponse()

            def close(self):
                pass

        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "guide.pdf"
            pdf_path.write_bytes(b"pdf-bytes")
            connections = []

            def make_connection(host, timeout):
                connection = FakeConnection(host, timeout)
                connections.append(connection)
                return connection

            with patch.object(parse_pdf.http.client, "HTTPSConnection", side_effect=make_connection):
                parse_pdf.upload_pdf_to_presigned_url("https://example.test/upload.pdf?signature=1", pdf_path)

            method, path, body, headers = connections[0].requests[0]
            self.assertEqual(method, "PUT")
            self.assertEqual(path, "/upload.pdf?signature=1")
            self.assertEqual(body, b"pdf-bytes")
            self.assertNotIn("Content-Type", headers)


class EmbedChunksTests(unittest.TestCase):
    def test_skips_needs_split_chunks_and_writes_vectors(self):
        class FakeModel:
            def encode(self, texts, **kwargs):
                return {"dense_vecs": [[0.1] * embed_chunks.EXPECTED_VECTOR_DIMENSION for _ in texts]}

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_path = temp_path / "parent_child_chunks.jsonl"
            output_dir = temp_path / "vectors"
            records = [
                {"child_id": "child_00001", "text": "正常文本", "chunk_type": "text"},
                {"child_id": "child_00002", "text": "很长表格", "chunk_type": "table", "needs_split": True},
            ]
            chunks_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")

            with patch.object(embed_chunks, "load_model", return_value=FakeModel()):
                vectors_path, total_count, success_count, skipped_count = embed_chunks.embed_chunks(
                    chunks_path=chunks_path,
                    output_dir=output_dir,
                    model_name="fake",
                    batch_size=2,
                    max_length=128,
                    use_fp16=False,
                )

            output_records = [json.loads(line) for line in vectors_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(total_count, 2)
            self.assertEqual(success_count, 1)
            self.assertEqual(skipped_count, 1)
            self.assertEqual(output_records[0]["child_id"], "child_00001")
            self.assertEqual(len(output_records[0]["vector"]), embed_chunks.EXPECTED_VECTOR_DIMENSION)


if __name__ == "__main__":
    unittest.main()
