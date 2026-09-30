# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from rank_bm25 import BM25Okapi

import app.main as main
from app.main import app
from app.rag import RAGEngine, parse_scope_terms, tokenize


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.engine = RAGEngine.__new__(RAGEngine)
        self.engine.rows = [
            {"id": 1, "text": "高血压诊断需要多次测量血压。", "chunk_index": 1, "source": "test"},
            {"id": 2, "text": "80岁以上老人血压目标可以放宽到150/90 mmHg以下。", "chunk_index": 2, "source": "test"},
            {"id": 3, "text": "阿司匹林用于部分心脑血管疾病二级预防。", "chunk_index": 3, "source": "test"},
        ]
        self.engine.by_id = {r["id"]: r for r in self.engine.rows}
        self.engine.bm25 = BM25Okapi([tokenize(r["text"]) for r in self.engine.rows])
        self.engine.min_vector_score = 0.55
        self.engine.reranker = None
        self.engine.rerank_n = 10

    def test_scope_gate_keeps_current_guide_for_blood_pressure(self):
        self.engine._vector_search = lambda _query, _limit=10: {
            1: {"id": 1, "vector_score": 0.80, "entity": self.engine.by_id[1]}
        }
        self.engine._bm25_search = lambda _query, _limit=10: {}
        self.assertTrue(self.engine.retrieve("高血压怎么诊断"))

    def test_scope_gate_does_not_apply_blood_pressure_guide_to_other_topics(self):
        self.engine._vector_search = lambda _query, _limit=10: {
            1: {"id": 1, "vector_score": 0.95, "entity": self.engine.by_id[1]}
        }
        self.engine._bm25_search = lambda _query, _limit=10: {
            1: {"id": 1, "bm25_score": 2.0, "entity": self.engine.by_id[1]}
        }
        self.assertEqual(self.engine.retrieve("感冒发烧怎么办"), [])

    def test_scope_terms_are_configurable(self):
        self.assertEqual(parse_scope_terms("糖尿病, 胰岛素"), ("糖尿病", "胰岛素"))
        self.assertEqual(parse_scope_terms("*"), ())

    def test_hybrid_retrieval_returns_bm25_scores(self):
        def fake_vector_search(_query, _limit=10):
            return {
                1: {"id": 1, "vector_score": 0.80, "entity": self.engine.by_id[1]},
                2: {"id": 2, "vector_score": 0.75, "entity": self.engine.by_id[2]},
            }

        self.engine._vector_search = fake_vector_search
        hits = self.engine.retrieve("80岁老人血压降到多少", top_k=3)

        self.assertTrue(hits)
        self.assertIn("rrf_score", hits[0])
        self.assertTrue(any(h["entity"]["chunk_index"] == 2 and h["bm25_score"] for h in hits))

    def test_low_score_without_bm25_is_filtered(self):
        self.engine._vector_search = lambda _query, _limit=10: {
            1: {"id": 1, "vector_score": 0.20, "entity": self.engine.by_id[1]}
        }
        self.engine._bm25_search = lambda _query, _limit=10: {}

        self.assertEqual(self.engine.retrieve("劳动合同怎么解除"), [])

    def test_reranker_can_change_candidate_order(self):
        def fake_vector_search(_query, _limit=10):
            return {
                1: {"id": 1, "vector_score": 0.90, "entity": self.engine.by_id[1]},
                2: {"id": 2, "vector_score": 0.80, "entity": self.engine.by_id[2]},
            }

        class FakeReranker:
            def predict(self, pairs, show_progress_bar=False):
                return [0.10, 0.95]

        self.engine._vector_search = fake_vector_search
        self.engine._bm25_search = lambda _query, _limit=10: {}
        self.engine.reranker = FakeReranker()

        hits = self.engine.retrieve("血压目标", top_k=2)
        self.assertEqual(hits[0]["entity"]["chunk_index"], 2)
        self.assertAlmostEqual(hits[0]["rerank_score"], 0.95)


class ApiRoleRoutingTest(unittest.TestCase):
    def test_legacy_non_doctor_role_is_unavailable_without_model_load(self):
        legacy_role = {"id": 3, "name": "律师", "persona_prompt": "旧版律师提示词"}
        client = TestClient(app)
        with patch.object(main.db, "get_role", return_value=legacy_role), patch.object(
            main, "get_engine", side_effect=AssertionError("Unexpected model load")
        ):
            response = client.post("/api/chat", json={"user_id": 1, "role_id": 3, "message": "劳动合同怎么解除"})
            self.assertEqual(response.status_code, 404)
            self.assertEqual(client.get("/api/history", params={"user_id": 1, "role_id": 3}).status_code, 404)
            self.assertEqual(client.delete("/api/history", params={"user_id": 1, "role_id": 3}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
