"""Focused tests for the public stages in app/offline_pipeline.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "app" / "offline_pipeline.py"


def load_pipeline():
    name = f"offline_pipeline_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeEmbeddingModel:
    init_args = None

    def __init__(self, *args, **kwargs):
        type(self).init_args = (args, kwargs)

    def encode(self, texts, **kwargs):
        return [FakeVector([float(index), 1.0]) for index, _ in enumerate(texts)]


class OfflinePipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = load_pipeline()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_import_is_lazy_for_large_optional_dependencies(self):
        for name in ("pymupdf", "pdfplumber", "paddleocr", "pymilvus",
                     "sentence_transformers"):
            self.assertNotIn(name, self.pipeline.__dict__)

    def test_parse_document_rejects_invalid_input_without_loading_parser(self):
        with self.assertRaisesRegex(ValueError, "有效PDF"):
            self.pipeline.parse_document(self.root / "missing.pdf")
        pdf = self.root / "note.pdf"
        pdf.write_bytes(b"not needed because engine is rejected first")
        with self.assertRaisesRegex(ValueError, "不支持"):
            self.pipeline.parse_document(pdf, engine="unknown")

    def test_parse_document_auto_uses_ocr_for_sparse_pdf(self):
        pdf = self.root / "scan.pdf"
        pdf.write_bytes(b"fake pdf for mocked parser")
        with patch.object(self.pipeline, "_pymupdf_text", return_value=("", 0)), \
             patch.object(self.pipeline, "_paddleocr_text", return_value="识别结果"):
            text, engine = self.pipeline.parse_document(pdf, engine="auto")
        self.assertEqual(("识别结果", "paddleocr"), (text, engine))

    def test_parse_document_auto_only_ocr_sparse_pages(self):
        pdf = self.root / "mixed.pdf"
        pdf.write_bytes(b"fake pdf for mocked parser")
        digital = "数字页正文" * 30
        with patch.object(self.pipeline, "_pymupdf_text",
                          return_value=(digital + "\n\f\n", len(digital) / 2)), \
             patch.object(self.pipeline, "_paddleocr_text", return_value="扫描页文字") as ocr:
            text, engine = self.pipeline.parse_document(pdf, engine="auto")
        self.assertEqual(f"{digital}\n\f\n扫描页文字", text)
        self.assertEqual("pymupdf+paddleocr", engine)
        ocr.assert_called_once_with(pdf, page_numbers=[1])

    def test_clean_and_split_are_public_stages(self):
        raw = "  标题  \n1\n这是第一段内容，包含  多余空格。\n" + "长句" * 80
        cleaned = self.pipeline.clean_text(raw)
        chunks = self.pipeline.split_chunks(cleaned, size=60, min_size=10)
        self.assertNotIn("\n1\n", f"\n{cleaned}\n")
        self.assertIn("包含 多余空格", cleaned)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) >= 10 for chunk in chunks))

    def test_split_chunks_merges_short_tail_instead_of_dropping_it(self):
        chunks = self.pipeline.split_chunks("甲" * 60 + "\n短尾", size=60, min_size=10)
        self.assertEqual(1, len(chunks))
        self.assertTrue(chunks[0].endswith("短尾"))
        self.assertEqual(["短文"], self.pipeline.split_chunks("短文", size=60, min_size=10))

    def test_embed_chunks_uses_local_model_and_writes_vectors(self):
        chunks = self.root / "data" / "chunks"
        models = self.root / "models" / "demo-bge"
        chunks.mkdir(parents=True)
        models.mkdir(parents=True)
        source = [{"source": "guide", "index": 0, "text": "片段一"},
                  {"source": "guide", "index": 1, "text": "片段二"}]
        (chunks / "guide.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in source),
            encoding="utf-8",
        )
        old_env = os.environ.get("RAG_EMBEDDING_DEVICE")
        os.environ["RAG_EMBEDDING_DEVICE"] = "cpu"
        try:
            result = self.pipeline.embed_chunks(
                self.root, model_name="demo-bge", model_factory=FakeEmbeddingModel
            )
        finally:
            if old_env is None:
                os.environ.pop("RAG_EMBEDDING_DEVICE", None)
            else:
                os.environ["RAG_EMBEDDING_DEVICE"] = old_env
        output = [json.loads(line) for line in
                  (self.root / "data" / "embeddings" / "guide.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        self.assertEqual(2, result[0]["dimension"])
        self.assertEqual([0.0, 1.0], output[0]["vector"])
        self.assertEqual("demo-bge", output[0]["embedding_model"])
        self.assertTrue(FakeEmbeddingModel.init_args[1]["local_files_only"])
        manifest = json.loads(
            (self.root / "data" / "embeddings" / "snapshot-manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(["guide.jsonl"], [item["name"] for item in manifest["files"]])
        self.assertEqual(
            self.pipeline._sha256(self.root / "data" / "embeddings" / "guide.jsonl"),
            manifest["files"][0]["sha256"],
        )

    def test_parser_status_tolerates_module_with_no_spec(self):
        with patch.dict(sys.modules, {"paddleocr": object()}):
            status = self.pipeline._parser_status()
        self.assertFalse(status["paddleocr"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
