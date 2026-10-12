import asyncio
import unittest


class FakeRAG:
    async def answer(self, question, top_k=5, mode="hybrid"):
        if "故障" in question:
            raise RuntimeError("simulated")
        return {
            "answer": "营业收入1379.58亿元，净利润281.95亿元。",
            "mode": "evidence",
            "retrieval_ms": 12.0,
            "generation_ms": 0.0,
            "total_ms": 12.0,
            "hits": [
                {"stock_code": "000001", "page": 5, "text": "营业收入1,379.58亿元，净利润281.95亿元", "score": 0.9},
                {"stock_code": "600036", "page": 7, "text": "其他材料", "score": 0.5},
            ],
        }


class EvaluationTests(unittest.TestCase):
    def test_question_bank_contains_exactly_ten_repeatable_questions(self):
        from evaluation import load_questions

        questions = load_questions()

        self.assertEqual(len(questions), 10)
        self.assertEqual(len({item["id"] for item in questions}), 10)
        self.assertTrue(all(item["reference_answer"] for item in questions))
        self.assertTrue(all(item["expected_facts"] for item in questions))

    def test_retrieval_metrics_use_first_relevant_rank_and_recall(self):
        from evaluation import score_retrieval

        question = {"stock_codes": ["000001"], "target_pages": [5, 25]}
        hits = [
            {"stock_code": "600036", "page": 7, "text": "x"},
            {"stock_code": "000001", "page": 5, "text": "y"},
            {"stock_code": "000001", "page": 25, "text": "z"},
        ]

        score = score_retrieval(hits, question)

        self.assertEqual(score, {"hit": 1, "recall": 1.0, "reciprocal_rank": 0.5})

    def test_retrieval_miss_has_zero_mrr(self):
        from evaluation import score_retrieval

        score = score_retrieval(
            [{"stock_code": "600036", "page": 7, "text": "x"}],
            {"stock_codes": ["000001"], "target_pages": [5]},
        )

        self.assertEqual(score["hit"], 0)
        self.assertEqual(score["reciprocal_rank"], 0.0)

    def test_matching_expected_fact_counts_as_relevant_on_an_alternative_page(self):
        from evaluation import score_retrieval

        score = score_retrieval(
            [{"stock_code": "000001", "page": 30, "text": "净利润281.95亿元"}],
            {
                "stock_codes": ["000001"],
                "target_pages": [5],
                "expected_facts": [["281.95"]],
            },
        )

        self.assertEqual(score, {"hit": 1, "recall": 1.0, "reciprocal_rank": 1.0})

    def test_fact_score_accepts_alternative_number_format(self):
        from evaluation import score_facts

        score = score_facts("营业收入为1,379.58亿元，净利润281.95亿元。", [["1379.58", "1,379.58"], ["281.95"]])

        self.assertEqual(score, {"matched": 2, "total": 2, "accuracy": 1.0})

    def test_evaluate_all_keeps_running_when_one_question_fails(self):
        from evaluation import evaluate_all

        questions = [
            {
                "id": 1,
                "question": "平安银行业绩？",
                "reference_answer": "营业收入1379.58亿元，净利润281.95亿元。",
                "stock_codes": ["000001"],
                "target_pages": [5],
                "expected_facts": [["1379.58"], ["281.95"]],
            },
            {
                "id": 2,
                "question": "故障问题",
                "reference_answer": "无",
                "stock_codes": ["000001"],
                "target_pages": [5],
                "expected_facts": [["无"]],
            },
        ]

        report = asyncio.run(evaluate_all(FakeRAG(), questions, top_k=3))

        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["results"][0]["fact_score"]["accuracy"], 1.0)
        self.assertEqual(report["results"][1]["status"], "error")
        self.assertEqual(report["summary"]["completed"], 1)


if __name__ == "__main__":
    unittest.main()
