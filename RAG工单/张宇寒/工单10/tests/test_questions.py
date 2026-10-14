import tempfile
import unittest
from pathlib import Path

from questions import load_questions


ROOT = Path(__file__).resolve().parents[1]


class QuestionMarkdownTests(unittest.TestCase):
    def test_real_question_file_contains_ten_complete_questions(self):
        questions = load_questions(ROOT / "eval_questions.md")
        self.assertEqual([item["id"] for item in questions], list(range(1, 11)))
        self.assertEqual(questions[0]["stocks"], ["000001"])
        self.assertEqual(questions[-1]["stocks"], ["600030", "600999", "601211"])
        self.assertEqual(questions[-1]["targets"]["601211"], [29])
        self.assertEqual(questions[0]["expected_facts"][0], ["1379.58", "1,379.58"])

    def test_missing_required_question_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.md"
            path.write_text("# questions\n\n## 1\n- Question: 只有问题", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reference"):
                load_questions(path)


if __name__ == "__main__":
    unittest.main()

