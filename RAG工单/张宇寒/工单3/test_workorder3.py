import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import requests

import app as web_app
from prepare import MinerUClient, build_chunks, get_or_create_batch
from rag import Chunk, LocalIndex, load_faiss_index, save_faiss_index, understand_query


ROOT = Path(__file__).resolve().parent


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")


class WorkOrder3Tests(unittest.TestCase):
    def test_right_click_launcher_starts_workorder_port(self):
        with patch("uvicorn.run") as run:
            web_app.run_server()
        run.assert_called_once_with(web_app.app, host="127.0.0.1", port=8013)

    def test_query_routes_each_company_to_its_own_document(self):
        self.assertEqual(
            understand_query("武汉力源信息技术股份有限公司本次发行股数是多少？")["document"],
            "prospectus_2",
        )
        self.assertEqual(
            understand_query("武汉兴图新科电子股份有限公司注册资本是多少？")["document"],
            "prospectus_1",
        )

    def test_chunking_keeps_document_and_company_metadata(self):
        chunks = build_chunks(
            [
                {
                    "page": 1,
                    "type": "text",
                    "text": "本次发行股数为示例数值。",
                    "document": "prospectus_2",
                    "company": "武汉力源信息技术股份有限公司",
                }
            ]
        )
        self.assertEqual(chunks[0].document, "prospectus_2")
        self.assertEqual(chunks[0].company, "武汉力源信息技术股份有限公司")

    def test_title_from_first_document_does_not_leak_into_second_document(self):
        chunks = build_chunks(
            [
                {"page": 1, "type": "title", "text": "第一份标题", "document": "prospectus_1", "company": "兴图新科"},
                {"page": 1, "type": "table", "text": "第一份表格", "document": "prospectus_1", "company": "兴图新科"},
                {"page": 1, "type": "text", "text": "第二份正文", "document": "prospectus_2", "company": "力源信息"},
            ]
        )
        second = next(chunk for chunk in chunks if chunk.document == "prospectus_2")
        self.assertEqual(second.title, "")

    def test_search_excludes_chunks_from_the_other_document(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="float32"))
        chunks = [
            Chunk("1", "兴图新科内容", "", 1, document="prospectus_1", company="兴图新科"),
            Chunk("2", "力源信息内容", "", 1, document="prospectus_2", company="力源信息"),
        ]
        hits = LocalIndex(index, chunks, FakeEncoder()).search(
            ["发行股数"], top_k=1, document="prospectus_2"
        )
        self.assertEqual(hits[0].chunk.document, "prospectus_2")

    def test_benchmark_contains_four_new_and_ten_existing_questions(self):
        questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(questions), 14)
        self.assertEqual(len({item["case_id"] for item in questions}), 14)
        counts = {
            document: sum(item["document"] == document for item in questions)
            for document in ("prospectus_1", "prospectus_2")
        }
        self.assertEqual(counts, {"prospectus_1": 10, "prospectus_2": 4})

    def test_upload_reopens_pdf_and_retries_after_connection_break(self):
        class JsonResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"code": 0, "data": {"batch_id": "batch-1", "file_urls": ["https://upload/1"]}}

        class UploadResponse:
            status_code = 200

            def raise_for_status(self):
                return None

        class Session:
            def __init__(self):
                self.put_calls = 0
                self.uploaded = b""

            def post(self, *_args, **_kwargs):
                return JsonResponse()

            def put(self, _url, data, **_kwargs):
                self.put_calls += 1
                if self.put_calls == 1:
                    data.read(3)
                    raise requests.ConnectionError("connection aborted")
                self.uploaded = data.read()
                return UploadResponse()

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "sample.pdf"
            pdf.write_bytes(b"complete-pdf")
            client = MinerUClient("token")
            client.session = Session()
            with patch("prepare.time.sleep", return_value=None):
                batch_id = client.submit(pdf, "doc", ((1, "1-1"),))
        self.assertEqual(batch_id, "batch-1")
        self.assertEqual(client.session.put_calls, 2)
        self.assertEqual(client.session.uploaded, b"complete-pdf")

    def test_download_resumes_partial_zip_and_extracts_it(self):
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as package:
            package.writestr("result_content_list.json", "[]")
        archive_bytes = archive_buffer.getvalue()
        split = 17

        class StreamResponse:
            def __init__(self, status_code, headers, chunks):
                self.status_code = status_code
                self.headers = headers
                self.chunks = chunks

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, _chunk_size):
                for chunk in self.chunks:
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk

        class Session:
            def __init__(self):
                self.headers = []

            def get(self, _url, **kwargs):
                self.headers.append(kwargs.get("headers", {}))
                if len(self.headers) == 1:
                    return StreamResponse(
                        200,
                        {"Content-Length": str(len(archive_bytes))},
                        [archive_bytes[:split], requests.exceptions.ChunkedEncodingError("broken")],
                    )
                return StreamResponse(
                    206,
                    {"Content-Range": f"bytes {split}-{len(archive_bytes) - 1}/{len(archive_bytes)}"},
                    [archive_bytes[split:]],
                )

        with tempfile.TemporaryDirectory() as directory:
            client = MinerUClient("token")
            client.session = Session()
            with patch("prepare.MINERU_DIR", Path(directory)), patch(
                "prepare.time.sleep", return_value=None
            ):
                paths = client.download([{"full_zip_url": "https://download/result"}], "doc")
            self.assertTrue((paths[0] / "result_content_list.json").exists())
            self.assertEqual(client.session.headers[1]["Range"], f"bytes={split}-")

    def test_saved_batch_is_reused_after_program_restart(self):
        class Client:
            def __init__(self):
                self.calls = 0

            def submit(self, *_args):
                self.calls += 1
                return "batch-saved"

        config = {
            "document": "prospectus_test",
            "pdf_path": Path("sample.pdf"),
            "page_ranges": ((1, "1-1"),),
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "prepare.MINERU_DIR", Path(directory)
        ):
            first = Client()
            self.assertEqual(get_or_create_batch(first, config), "batch-saved")
            second = Client()
            self.assertEqual(get_or_create_batch(second, config), "batch-saved")
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)

    def test_faiss_index_round_trips_through_unicode_path(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文目录" / "索引.faiss"
            path.parent.mkdir()
            save_faiss_index(index, path)
            loaded = load_faiss_index(path)
        self.assertEqual(loaded.ntotal, 1)
        scores, ids = loaded.search(np.asarray([[1.0, 0.0]], dtype="float32"), 1)
        self.assertEqual(int(ids[0][0]), 0)
        self.assertAlmostEqual(float(scores[0][0]), 1.0)


if __name__ == "__main__":
    unittest.main()
