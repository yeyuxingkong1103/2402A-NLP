import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hybrid_search import compact_hit, reciprocal_rank_fusion


class HybridSearchTests(unittest.TestCase):
    def test_rrf_promotes_hits_present_in_both_retrievers(self):
        dense = [{"id": "a"}, {"id": "b"}]
        bm25 = [{"id": "b"}, {"id": "c"}]

        fused = reciprocal_rank_fusion([dense, bm25], k=60)

        self.assertEqual(fused[0]["id"], "b")

    def test_rrf_merges_fields_scores_and_ranks(self):
        dense = [{"id": "a", "text": "dense text", "dense_score": 0.8}]
        bm25 = [{"id": "a", "text": "bm25 text", "bm25_score": 9.0, "document_id": "doc"}]

        fused = reciprocal_rank_fusion([dense, bm25], k=60)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["text"], "dense text")
        self.assertEqual(fused[0]["dense_score"], 0.8)
        self.assertEqual(fused[0]["bm25_score"], 9.0)
        self.assertEqual(fused[0]["document_id"], "doc")
        self.assertEqual(fused[0]["dense_rank"], 1)
        self.assertEqual(fused[0]["bm25_rank"], 1)
        self.assertGreater(fused[0]["fusion_score"], 0)

    def test_rrf_handles_empty_retriever_results(self):
        fused = reciprocal_rank_fusion([[{"id": "a"}], []], k=60)

        self.assertEqual([item["id"] for item in fused], ["a"])

    def test_compact_hit_preserves_rerank_score(self):
        compacted = compact_hit(
            {
                "id": "a",
                "text": "正文",
                "section_path": "section",
                "chunk_type": "text",
                "rerank_score": 0.95,
            }
        )

        self.assertEqual(compacted["rerank_score"], 0.95)

        dense = [{"id": "a"}, {"id": "a"}, {"id": "b"}]

        fused = reciprocal_rank_fusion([dense], k=60)

        self.assertEqual([item["id"] for item in fused], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
