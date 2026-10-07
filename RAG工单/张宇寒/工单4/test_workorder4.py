import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from prepare import build_chunks, load_figure_chunks
from rag import Chunk, LocalIndex, load_faiss_index, save_faiss_index, understand_query


ROOT = Path(__file__).resolve().parent


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")


class WorkOrder4Tests(unittest.TestCase):
    def test_faiss_index_round_trips_through_unicode_path(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文目录" / "索引.faiss"
            path.parent.mkdir()
            save_faiss_index(index, path)
            loaded = load_faiss_index(path)
        self.assertEqual(loaded.ntotal, 1)

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

    def test_chart_question_retrieves_figure_evidence_before_generic_text(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
        chunks = [
            Chunk("generic", "公司介绍", "", 1, document="prospectus_2", company="力源信息"),
            Chunk("figure", "汽车电子14.0%，IC卡-2.0%", "2008年中国IC市场应用结构与增长图", 72, content_type="figure", document="prospectus_2", company="力源信息"),
        ]
        hits = LocalIndex(index, chunks, FakeEncoder()).search(
            ["2008年中国IC市场应用结构与增长图中哪个行业增长最快？"],
            top_k=1,
            document="prospectus_2",
        )
        self.assertEqual(hits[0].chunk.chunk_id, "figure")

    def test_benchmark_contains_six_pdf2_and_ten_pdf1_questions(self):
        questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(questions), 16)
        self.assertEqual(len({item["case_id"] for item in questions}), 16)
        counts = {
            document: sum(item["document"] == document for item in questions)
            for document in ("prospectus_1", "prospectus_2")
        }
        self.assertEqual(counts, {"prospectus_1": 10, "prospectus_2": 6})

    def test_figure_evidence_preserves_real_pdf_pages_and_full_lists(self):
        figures = load_figure_chunks(ROOT / "figure_facts.json")
        self.assertEqual({chunk.page for chunk in figures}, {39, 72})
        organization = next(chunk for chunk in figures if chunk.page == 39)
        market = next(chunk for chunk in figures if chunk.page == 72)
        self.assertEqual(organization.content_type, "figure")
        self.assertEqual(organization.document, "prospectus_2")
        for fact in ("渠道销售部", "电话及网络销售部", "大客户销售部", "国际贸易部", "珠海销售处", "深圳销售处", "北京销售处", "武汉销售处", "广州销售处", "成都销售处"):
            self.assertIn(fact, organization.text)
        self.assertIn("汽车电子", market.text)
        self.assertIn("14.0%", market.text)
        self.assertIn("IC卡", market.text)
        self.assertIn("-2.0%", market.text)


if __name__ == "__main__":
    unittest.main()
