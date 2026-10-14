import unittest

import numpy as np

from evaluator import compare_models, evaluate_retrieval


class StaticModel:
    def encode(self, texts, adapted=False):
        mapping = {
            "q1": [1.0, 0.0],
            "q2": [0.0, 1.0],
            "d1": [1.0, 0.0],
            "d2": [0.0, 1.0],
            "noise": [0.7, 0.7],
        }
        vectors = np.array([mapping[text] for text in texts], dtype=float)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


RECORDS = [
    {"question_id": 1, "query": "q1", "positive_id": "a"},
    {"question_id": 2, "query": "q2", "positive_id": "b"},
]

DOCUMENTS = [
    {"chunk_id": "a", "text": "d1"},
    {"chunk_id": "b", "text": "d2"},
    {"chunk_id": "c", "text": "noise"},
]


class EvaluatorTests(unittest.TestCase):
    def test_perfect_retrieval_has_literal_recall_and_mrr(self):
        result = evaluate_retrieval(StaticModel(), RECORDS, DOCUMENTS, top_k=1)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual([item["rank"] for item in result["queries"]], [1, 1])
        self.assertGreater(result["similarity_gap"], 0)

    def test_comparison_reports_before_after_delta(self):
        result = compare_models(StaticModel(), RECORDS, DOCUMENTS, top_k=1)
        self.assertEqual(result["before"]["recall_at_k"], 1.0)
        self.assertEqual(result["after"]["recall_at_k"], 1.0)
        self.assertEqual(result["delta"]["mrr"], 0.0)

    def test_invalid_top_k_and_unknown_positive_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Top K"):
            evaluate_retrieval(StaticModel(), RECORDS, DOCUMENTS, top_k=0)
        bad = [{"question_id": 9, "query": "q1", "positive_id": "missing"}]
        with self.assertRaisesRegex(ValueError, "正例"):
            evaluate_retrieval(StaticModel(), bad, DOCUMENTS, top_k=1)


if __name__ == "__main__":
    unittest.main()
