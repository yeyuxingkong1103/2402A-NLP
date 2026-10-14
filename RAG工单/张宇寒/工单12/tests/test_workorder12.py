import json
import unittest
import uuid
from pathlib import Path

from evaluation import evaluate_result
from knowledge_graph import KnowledgeGraph
from questions import QUESTIONS
from retrieval import RetrievalEngine


class QuestionCatalogTests(unittest.TestCase):
    def test_contains_exactly_the_16_required_questions(self):
        self.assertEqual(16, len(QUESTIONS))
        self.assertEqual(
            {1, 2, 3, 4, 5, 6, 33, 34, 95, 207, 260, 531, 543, 793, 795, 957},
            {item["id"] for item in QUESTIONS},
        )


class KnowledgeGraphTests(unittest.TestCase):
    def test_duplicate_triples_are_merged_and_incremental_update_counts_only_new_data(self):
        graph = KnowledgeGraph()
        triple = ("武汉力源信息技术股份有限公司", "HAS_DEPARTMENT", "销售部")
        first = graph.add_triples([triple, triple], source_id="chunk-1")
        second = graph.add_triples([triple], source_id="chunk-1")
        self.assertEqual({"nodes_added": 2, "edges_added": 1}, first)
        self.assertEqual({"nodes_added": 0, "edges_added": 0}, second)
        self.assertEqual(1, len(graph.edges))

    def test_graph_round_trip_preserves_relations(self):
        graph = KnowledgeGraph()
        graph.add_triples([("大客户销售部", "HAS_SALES_OFFICE", "北京销售处")], "p2-5")
        folder = Path(__file__).parent / f".tmp-{uuid.uuid4().hex}"
        path = folder / "graph.json"
        try:
            graph.save(path)
            loaded = KnowledgeGraph.load(path)
        finally:
            if path.exists():
                path.unlink()
            if folder.exists():
                folder.rmdir()
        self.assertEqual(["北京销售处"], [n["name"] for n in loaded.neighbors("大客户销售部")])


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id": "noise", "document": "prospectus_2", "page": 1, "title": "概览", "text": "公司经营电子元器件分销业务。"},
            {"chunk_id": "fact", "document": "prospectus_2", "page": 39, "title": "组织结构", "text": "大客户销售部下设北京销售处、深圳销售处、广州销售处、成都销售处、珠海销售处和武汉销售处，共6个销售处。"},
        ]
        graph = KnowledgeGraph()
        graph.add_triples([("大客户销售部", "HAS_SALES_OFFICE", "北京销售处")], "fact")
        self.engine = RetrievalEngine(self.chunks, graph)

    def test_lightrag_retrieval_exposes_graph_path_and_evidence(self):
        result = self.engine.search("大客户销售部有几个销售处？", mode="lightrag", strategy="hybrid", top_k=2)
        self.assertEqual("fact", result["contexts"][0]["chunk_id"])
        self.assertIn("大客户销售部", result["matched_entities"])
        self.assertTrue(result["graph_paths"])

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.search("问题", mode="unknown")


class EvaluationTests(unittest.TestCase):
    def test_context_recall_counts_independent_expected_facts(self):
        question = {"expected_facts": [["汽车电子"], ["IC卡", "IC 卡"]], "question": "增长情况？"}
        result = {"contexts": [{"text": "汽车电子增长最快，IC卡出现负增长。"}], "answer": "汽车电子增长最快，IC卡负增长。"}
        scores = evaluate_result(question, result)
        self.assertEqual(1.0, scores["context_recall"])
        self.assertEqual(1.0, scores["faithfulness"])

    def test_empty_context_has_zero_scores_instead_of_crashing(self):
        question = {"expected_facts": [["程家明"]], "question": "法定代表人是谁？"}
        scores = evaluate_result(question, {"contexts": [], "answer": ""})
        self.assertEqual(0.0, scores["context_precision"])
        self.assertEqual(0.0, scores["context_recall"])


if __name__ == "__main__":
    unittest.main()
