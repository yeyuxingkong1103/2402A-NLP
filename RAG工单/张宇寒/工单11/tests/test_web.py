from pathlib import Path
import unittest

from fastapi import HTTPException

import app as web


def run(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("测试接口出现了未预期的异步等待")


class WebTests(unittest.TestCase):
    def test_status_exposes_six_steps_and_before_after_metrics(self):
        result = run(web.status())
        self.assertTrue(result["ready"])
        self.assertEqual(len(result["steps"]), 6)
        self.assertIn("before", result["evaluation"])
        self.assertIn("after", result["evaluation"])

    def test_search_returns_ranked_financial_chunks(self):
        result = run(web.search(web.SearchRequest(query="平安银行营业收入", top_k=2, mode="after")))
        self.assertEqual(len(result["hits"]), 2)
        self.assertGreaterEqual(result["hits"][0]["score"], result["hits"][1]["score"])
        self.assertIn("text", result["hits"][0])

    def test_empty_search_is_rejected(self):
        with self.assertRaises(HTTPException) as captured:
            run(web.search(web.SearchRequest(query="   ", top_k=2)))
        self.assertEqual(captured.exception.status_code, 400)

    def test_page_has_training_steps_metrics_and_loss_chart(self):
        page = (Path(__file__).parents[1] / "index.html").read_text(encoding="utf-8")
        for marker in ("stepGrid", "metricGrid", "lossChart", "exampleTable", "searchResults"):
            self.assertIn(f'id="{marker}"', page)
        self.assertIn("工单11", page)
        self.assertEqual(web.PORT, 8022)


if __name__ == "__main__":
    unittest.main()
