import json
import tempfile
from pathlib import Path
import unittest

from train import TrainingConfig, run_training


TRAIN = [
    {"question_id": 1, "query": "甲公司营业收入", "positive": "甲公司营业收入100亿元", "negative": "甲公司员工500人", "positive_id": "a", "negative_id": "b"},
    {"question_id": 2, "query": "乙公司净利润", "positive": "乙公司净利润20亿元", "negative": "乙公司风险管理", "positive_id": "c", "negative_id": "d"},
]
EVALUATION = [
    {"question_id": 3, "query": "甲公司收入", "positive": "甲公司营业收入100亿元", "negative": "乙公司风险管理", "positive_id": "a", "negative_id": "d"},
]
DOCUMENTS = [
    {"chunk_id": "a", "text": "甲公司营业收入100亿元"},
    {"chunk_id": "b", "text": "甲公司员工500人"},
    {"chunk_id": "c", "text": "乙公司净利润20亿元"},
    {"chunk_id": "d", "text": "乙公司风险管理"},
]


class TrainingPipelineTests(unittest.TestCase):
    def test_training_writes_model_and_before_after_report(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            output = Path(folder)
            report = run_training(
                TRAIN,
                EVALUATION,
                DOCUMENTS,
                output,
                TrainingConfig(dimension=3, epochs=5, top_k=2),
            )
            self.assertTrue((output / "domain_embedding.pkl").is_file())
            self.assertTrue((output / "evaluation_report.json").is_file())
            saved = json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))
            self.assertIn("before", saved)
            self.assertIn("after", saved)
            self.assertEqual(len(report["training"]["loss_history"]), 5)
            self.assertEqual(report["training"]["loss_name"], "Triplet Loss")


if __name__ == "__main__":
    unittest.main()
