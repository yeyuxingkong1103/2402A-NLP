"""只用标准库的检索指标测试，不加载模型或读取API配置。"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1] / "app/evaluate.py"
spec = importlib.util.spec_from_file_location("retrieval_metrics", path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class RetrievalMetricsTests(unittest.TestCase):
    def test_known_ranking_and_denominator(self):
        score = metrics.retrieval_score_case([99, 21, 19, 3], [19, 21], 4)
        self.assertEqual(score["hit_at_k"], 1)
        self.assertEqual(score["recall_at_k"], 1)
        self.assertEqual(score["precision_at_k"], 0.5)
        self.assertEqual(score["mrr"], 0.5)

    def test_no_hit_is_zero_for_answerable_case(self):
        score = metrics.retrieval_score_case([], [19], 4)
        self.assertTrue(all(score[name] == 0 for name in metrics.RETRIEVAL_METRICS))

    def test_out_of_scope_is_separate(self):
        empty = metrics.retrieval_score_case([], [], 4)
        self.assertTrue(all(empty[name] is None for name in metrics.RETRIEVAL_METRICS))
        self.assertEqual(empty["out_of_scope_correct"], 1)
        self.assertEqual(metrics.retrieval_score_case([1], [], 4)["out_of_scope_correct"], 0)

    def test_duplicate_does_not_inflate_hits(self):
        score = metrics.retrieval_score_case([19, 19, 21], [19, 21], 4)
        self.assertEqual(score["precision_at_k"], 0.5)
        self.assertEqual(score["matched"], [19, 21])

    def test_failure_and_out_of_scope_do_not_inflate_mean(self):
        rows = [
            {"variant": "hybrid", "status": "ok", "elapsed_ms": 10, "scores": metrics.retrieval_score_case([], [19])},
            {"variant": "hybrid", "status": "ok", "elapsed_ms": 10, "scores": metrics.retrieval_score_case([], [])},
            {"variant": "hybrid", "status": "failed", "elapsed_ms": 10, "scores": None},
        ]
        summary = metrics.retrieval_summarize(rows)["hybrid"]
        self.assertEqual(summary["mrr"], 0)
        self.assertEqual(summary["mrr_count"], 1)
        self.assertEqual(summary["out_of_scope_correct"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            metrics.retrieval_score_case([], [], 0)

    def test_comparison_refuses_unavailable_reranker(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cases = folder / "cases.json"
            cases.write_text(json.dumps([{"id": "a", "question": "血压", "relevant_chunk_indexes": [1]}]), encoding="utf-8")
            args = SimpleNamespace(cases=cases, compare_rerank=True, rerank=False, top_k=4,
                                   project_root=folder, output=folder / "result")
            engine = Mock(rerank_state="unavailable", reranker=None)
            with patch.object(metrics, "build_retrieval_engine", return_value=engine):
                self.assertEqual(metrics.run_retrieval(args), 1)
            engine.retrieve.assert_not_called()
            report = json.loads((folder / "result_failed.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "initialization_failed")
            self.assertEqual(report["summary"], {})

    def test_retrieval_failure_remains_failure_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cases = folder / "cases.json"
            cases.write_text(json.dumps([{"id": "a", "question": "血压", "relevant_chunk_indexes": [1]}]), encoding="utf-8")
            args = SimpleNamespace(cases=cases, compare_rerank=False, rerank=False, top_k=4,
                                   project_root=folder, output=folder / "result")
            engine = Mock(rerank_state="disabled", reranker=None, rows=[{"source": "guide", "chunk_index": 1}])
            engine.retrieve.side_effect = RuntimeError("private diagnostic detail")
            with patch.object(metrics, "build_retrieval_engine", return_value=engine):
                self.assertEqual(metrics.run_retrieval(args), 1)
            report = json.loads((folder / "result_failed.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed_with_failures")
            self.assertIsNone(report["summary"]["hybrid"]["mrr"])
            self.assertIsNone(report["results"][0]["scores"])
            self.assertNotIn("private diagnostic detail", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
