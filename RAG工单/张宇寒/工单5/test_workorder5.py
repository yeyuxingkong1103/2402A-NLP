import json
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from prepare import build_chunks, load_figure_chunks
from rag import Chunk, LocalIndex, understand_query
from conversation import ConversationManager
from app import AskRequest, ask


ROOT = Path(__file__).resolve().parent


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")


class WorkOrder5Tests(unittest.TestCase):
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

    def test_benchmark_contains_seven_pdf2_and_ten_pdf1_questions(self):
        questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(questions), 17)
        self.assertEqual(len({item["case_id"] for item in questions}), 17)
        counts = {
            document: sum(item["document"] == document for item in questions)
            for document in ("prospectus_1", "prospectus_2")
        }
        self.assertEqual(counts, {"prospectus_1": 10, "prospectus_2": 7})
        sales = next(item for item in questions if item["case_id"] == "p2-7")
        self.assertIn("哪个销售部的销售处最多", sales["question"])
        self.assertEqual(len(sales["expected_facts"]), 7)

    def test_followups_keep_company_and_switch_for_elliptical_question(self):
        sessions = ConversationManager()
        first = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"
        self.assertEqual(sessions.resolve(first, "a"), first)
        sessions.remember("a", first)
        second = sessions.resolve("他参与的哪个工程荣获了国家科技进步一等奖？", "a")
        self.assertIn("武汉兴图新科电子股份有限公司", second)
        sessions.remember("a", second)
        third = sessions.resolve("这个公司的法定代表人是谁？", "a")
        self.assertEqual(third, "武汉兴图新科电子股份有限公司的法定代表人是谁？")
        sessions.remember("a", third)
        fourth = sessions.resolve("那武汉力源信息技术股份有限公司呢？", "a")
        self.assertEqual(fourth, "武汉力源信息技术股份有限公司的法定代表人是谁？")

    def test_new_session_does_not_inherit_company(self):
        sessions = ConversationManager()
        sessions.remember("a", "武汉兴图新科电子股份有限公司的法定代表人是谁？")
        with self.assertRaisesRegex(ValueError, "公司"):
            sessions.resolve("这个公司的法定代表人是谁？", "b")

    def test_explicit_company_overrides_previous_company(self):
        sessions = ConversationManager()
        sessions.remember("a", "武汉兴图新科电子股份有限公司的法定代表人是谁？")
        question = "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？"
        self.assertEqual(sessions.resolve(question, "a"), question)

    def test_other_is_not_mistaken_for_pronoun(self):
        sessions = ConversationManager()
        sessions.remember("a", "武汉兴图新科电子股份有限公司的法定代表人是谁？")
        self.assertEqual(
            sessions.resolve("其他关联方有哪些？", "a"),
            "武汉兴图新科电子股份有限公司：其他关联方有哪些？",
        )

    def test_company_switch_after_short_name_keeps_previous_intent(self):
        sessions = ConversationManager()
        sessions.remember("a", "兴图新科的法定代表人是谁？")
        self.assertEqual(
            sessions.resolve("那力源信息呢？", "a"),
            "武汉力源信息技术股份有限公司的法定代表人是谁？",
        )

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


class AskRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_returns_session_and_rewrites_followup_before_retrieval(self):
        seen = []

        class FakeRAG:
            async def ask(self, question):
                seen.append(question)
                return {"answer": "依据文档的答案", "citations": [], "total_ms": 1}

        with patch("app.get_rag", return_value=FakeRAG()):
            first = await ask(AskRequest(question="武汉兴图新科电子股份有限公司的法定代表人是谁？"))
            second = await ask(AskRequest(question="那武汉力源信息技术股份有限公司呢？", session_id=first["session_id"]))
        self.assertEqual(seen[1], "武汉力源信息技术股份有限公司的法定代表人是谁？")
        self.assertEqual(second["session_id"], first["session_id"])
        self.assertEqual(second["resolved_question"], seen[1])


if __name__ == "__main__":
    unittest.main()
