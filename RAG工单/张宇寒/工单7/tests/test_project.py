import json
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ProjectDeliveryTests(unittest.TestCase):
    def test_env_leaves_deepseek_key_blank(self):
        values = dict(
            line.split("=", 1)
            for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        )

        self.assertIn("DEEPSEEK_API_KEY", values)
        self.assertEqual(values["DEEPSEEK_API_KEY"], "")

    def test_run_server_uses_port_8018(self):
        from app import run_server

        with patch("app.uvicorn.run") as run:
            run_server()

        self.assertEqual(run.call_args.kwargs["port"], 8018)
        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")

    def test_generated_files_and_secrets_are_git_ignored(self):
        patterns = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

        self.assertTrue({".env", "data/", "evaluation_report.json"}.issubset(patterns))

    def test_question_schema_is_complete(self):
        questions = json.loads((ROOT / "evaluation_questions.json").read_text(encoding="utf-8"))
        required = {"id", "question", "reference_answer", "targets", "expected_facts"}

        self.assertEqual(len(questions), 10)
        self.assertTrue(all(required.issubset(item) for item in questions))

    def test_readme_explains_prepare_start_and_export(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("prepare.py", readme)
        self.assertIn("app.py", readme)
        self.assertIn("8018", readme)
        self.assertIn("JSON", readme)


if __name__ == "__main__":
    unittest.main()
