import json
import unittest
from pathlib import Path

import faiss
import numpy as np

from prepare import build_chunks
from rag import Chunk, LocalIndex, understand_query


ROOT = Path(__file__).resolve().parent


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")


class WorkOrder3Tests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
