import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.pressure_test import jmeter_report, main, summarize_jtl


class PressureJtlTest(unittest.TestCase):
    fieldnames = ["timeStamp", "elapsed", "label", "responseCode", "success",
                  "failureMessage", "grpThreads", "URL"]

    def write_jtl(self, directory: Path, name: str, rows: list[dict]) -> Path:
        path = directory / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, started: int, elapsed: int, success="true", code="200") -> dict:
        return {"timeStamp": started, "elapsed": elapsed, "label": "RAG_CHAT",
                "responseCode": code, "success": success, "failureMessage": "",
                "grpThreads": 2, "URL": "http://127.0.0.1:8000/api/chat"}

    def test_summarize_uses_observation_window_and_nearest_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_jtl(Path(temp), "run.jtl",
                                  [self.row(1000, 1000), self.row(1000, 2000),
                                   self.row(3000, 3000)])
            report = summarize_jtl(path)
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["successful_qps"], 0.6)
        self.assertEqual(report["p50_latency_ms"], 2000)
        self.assertEqual(report["p95_latency_ms"], 3000)

    def test_assertion_failure_disables_successful_qps(self):
        with tempfile.TemporaryDirectory() as temp:
            failed = self.row(1000, 1000, success="false")
            failed["failureMessage"] = "sources为空"
            path = self.write_jtl(Path(temp), "failed.jtl", [failed])
            report = summarize_jtl(path)
        self.assertEqual(report["failures"], 1)
        self.assertIsNone(report["successful_qps"])
        self.assertEqual(report["failure_counts"], {"sources为空": 1})

    def test_multiple_jtl_runs_stay_separate_and_cli_writes_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.write_jtl(root, "c1.jtl", [self.row(1000, 1000)])
            second = self.write_jtl(root, "c2.jtl", [self.row(1000, 1000),
                                                       self.row(1000, 1000)])
            report = jmeter_report([first, second], "RAG_CHAT")
            self.assertEqual([run["samples"] for run in report["runs"]], [1, 2])
            output = root / "report"
            self.assertEqual(main(["--jtl", str(first), str(second),
                                   "--output", str(output)]), 0)
            saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved["samples"], 3)
            self.assertTrue(output.with_suffix(".md").exists())


if __name__ == "__main__":
    unittest.main()
