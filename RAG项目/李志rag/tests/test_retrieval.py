import unittest
from unittest.mock import patch

from app.rag_main import encode_query, reciprocal_rank_fusion, rerank_candidates


class RetrievalTests(unittest.TestCase):
    def test_query_embedding_uses_single_clean_question(self):
        with patch("app.rag_main.encode_documents", return_value=[[0.1, 0.2]]) as encode:
            vector = encode_query("  什么是血压？  ")
        self.assertEqual(vector, [0.1, 0.2])
        encode.assert_called_once_with(["什么是血压？"])

    def test_rrf_rewards_multiple_routes(self):
        dense = [
            {"chunk_id": "a", "text": "A", "document_id": 1, "source": "x"},
            {"chunk_id": "b", "text": "B", "document_id": 1, "source": "x"},
        ]
        sparse = [
            {"chunk_id": "b", "text": "B", "document_id": 1, "source": "x"},
            {"chunk_id": "c", "text": "C", "document_id": 2, "source": "y"},
        ]
        fused = reciprocal_rank_fusion([dense, sparse])
        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertEqual({item["chunk_id"] for item in fused}, {"a", "b", "c"})

    def test_rerank_can_keep_rrf_order(self):
        candidates = [{"chunk_id": "a", "score": 0.2}, {"chunk_id": "b", "score": 0.1}]
        with patch("app.rag_main.settings.rerank_enabled", False):
            result = rerank_candidates("问题", candidates, 1)
        self.assertEqual(result[0]["chunk_id"], "a")


if __name__ == "__main__":
    unittest.main()
