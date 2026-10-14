import subprocess
import unittest
from pathlib import Path

from dotenv import dotenv_values

from corpus import REPORTS
from graph_rag import DATA_DIR, GraphRAG
from questions import load_questions


ROOT = Path(__file__).resolve().parents[1]


class RealDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = GraphRAG.load(DATA_DIR)
        cls.questions = load_questions(ROOT / "eval_questions.md")

    def test_real_index_contains_all_nine_companies_and_graph_types(self):
        self.assertEqual({item.stock_code for item in self.rag.chunks}, set(REPORTS))
        self.assertGreater(len(self.rag.chunks), 1000)
        self.assertTrue({"company", "report", "year", "metric", "chunk"}.issubset({node["type"] for node in self.rag.graph["nodes"]}))

    def test_all_fixed_questions_retrieve_each_named_company_and_a_subgraph(self):
        for item in self.questions:
            with self.subTest(question=item["id"]):
                result = self.rag.search(item["question"], top_k=max(5, len(item["stocks"])))
                found = {hit["stock_code"] for hit in result["hits"]}
                self.assertTrue(set(item["stocks"]).issubset(found), (item["id"], found))
                self.assertTrue(result["graph"]["nodes"])
                self.assertTrue(result["graph"]["edges"])

    def test_delivery_env_is_blank_and_generated_data_is_git_ignored(self):
        values = dotenv_values(ROOT / ".env")
        self.assertEqual(values.get("DEEPSEEK_API_KEY"), "")
        if not (ROOT / ".git").is_dir():
            return
        result = subprocess.run(
            ["git", "check-ignore", "data/chunks.json", ".superpowers/progress.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()), {"data/chunks.json", ".superpowers/progress.md"})


if __name__ == "__main__":
    unittest.main()

