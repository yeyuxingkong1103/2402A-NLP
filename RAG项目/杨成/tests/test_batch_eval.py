import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import batch_eval


class BatchEvalTests(unittest.TestCase):
    def test_build_eval_record_success_keeps_required_fields(self):
        dataset_item = {"question": "问题", "ground_truth": "标准答案"}
        retrieved = [
            {"text": "上下文1", "section_path": "章节1"},
            {"text": "上下文2", "section_path": "章节2"},
        ]
        result = {
            "answer": "回答 [章节1]",
            "has_citation": True,
            "latency": {"total_ms": 123},
        }

        record = batch_eval.build_eval_record(dataset_item, result, retrieved)

        self.assertEqual(record["question"], "问题")
        self.assertEqual(record["ground_truth"], "标准答案")
        self.assertEqual(record["answer"], "回答 [章节1]")
        self.assertEqual(record["retrieved_contexts"], ["上下文1", "上下文2"])
        self.assertEqual(record["retrieved_sources"], ["章节1", "章节2"])
        self.assertEqual(record["latency_ms"], 123)
        self.assertTrue(record["has_citation"])

    def test_build_error_record_keeps_error_and_continues_shape(self):
        dataset_item = {"question": "问题", "ground_truth": "标准答案"}

        record = batch_eval.build_error_record(dataset_item, RuntimeError("timeout"), 456)

        self.assertEqual(record["question"], "问题")
        self.assertEqual(record["ground_truth"], "标准答案")
        self.assertEqual(record["answer"], "")
        self.assertEqual(record["retrieved_contexts"], [])
        self.assertEqual(record["retrieved_sources"], [])
        self.assertEqual(record["latency_ms"], 456)
        self.assertFalse(record["has_citation"])
        self.assertIn("timeout", record["error"])


if __name__ == "__main__":
    unittest.main()
