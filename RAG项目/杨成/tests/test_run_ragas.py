import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragas


class RunRagasTests(unittest.TestCase):
    def test_to_ragas_rows_maps_required_columns(self):
        results = [
            {
                "question": "问题",
                "answer": "回答",
                "retrieved_contexts": ["上下文"],
                "ground_truth": "标准答案",
            }
        ]

        rows = run_ragas.to_ragas_rows(results)

        self.assertEqual(
            rows,
            [
                {
                    "user_input": "问题",
                    "response": "回答",
                    "retrieved_contexts": ["上下文"],
                    "reference": "标准答案",
                }
            ],
        )

    def test_build_output_keeps_scores_and_averages(self):
        source = [
            {"question": "问题1", "answer": "回答1"},
            {"question": "问题2", "answer": "回答2"},
        ]
        scores = [
            {"faithfulness": 1.0, "answer_relevancy": 0.5, "context_recall": 0.0},
            {"faithfulness": 0.0, "answer_relevancy": 1.0, "context_recall": 1.0},
        ]

        output = run_ragas.build_output(source, scores)

        self.assertEqual(output["averages"]["faithfulness"], 0.5)
        self.assertEqual(output["averages"]["answer_relevancy"], 0.75)
        self.assertEqual(output["averages"]["context_recall"], 0.5)
        self.assertEqual(output["results"][0]["question"], "问题1")
        self.assertEqual(output["results"][0]["faithfulness"], 1.0)


if __name__ == "__main__":
    unittest.main()
