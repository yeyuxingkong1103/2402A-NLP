"""不调用付费 API 的评测工具测试；分数来自桩，不能作为系统质量报告。"""
import asyncio
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "app/evaluate.py"
spec = importlib.util.spec_from_file_location("evaluation_tool", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class FakeMetric:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    async def single_turn_ascore(self, sample, timeout):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def cases_file(self, cases):
        path = self.root / "cases.json"
        path.write_text(json.dumps(cases), encoding="utf-8")
        return path

    def test_duplicate_question_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            evaluation.load_ragas_cases(self.cases_file([{"id": "x", "question": "a"}, {"id": "x", "question": "b"}]))

    def test_reference_requires_source(self):
        with self.assertRaisesRegex(ValueError, "reference_sources"):
            evaluation.load_ragas_cases(self.cases_file([{"id": "x", "question": "a", "reference": "正确答案"}]))

    def test_reference_must_match_existing_chunk(self):
        (self.root / "chunks.jsonl").write_text(json.dumps({"index": 1, "text": "真实指南原文"}), encoding="utf-8")
        cases = [{"id": "a", "reference_sources": [{"path": "chunks.jsonl", "chunk_index": 1, "quote": "指南原文"}]}]
        self.assertEqual(evaluation.validate_references(cases, self.root), 1)
        cases[0]["reference_sources"][0]["quote"] = "凭空补造的引用"
        with self.assertRaisesRegex(ValueError, "不符"):
            evaluation.validate_references(cases, self.root)

    def test_nonfinite_score_is_excluded_not_zero(self):
        row = evaluation.new_row({"id": "a", "question": "q", "reference": None}, "hybrid")
        row["metrics"]["faithfulness"] = evaluation.metric_result(math.nan)
        summary = evaluation.aggregate([row])["hybrid"]["faithfulness"]
        self.assertIsNone(summary["mean"])
        self.assertEqual(summary["excluded"], 1)

    def test_skipped_and_failure_do_not_reduce_mean(self):
        rows = [evaluation.new_row({"id": str(i), "question": "q", "reference": None}, "hybrid") for i in range(3)]
        rows[0]["metrics"]["faithfulness"] = evaluation.metric_result(0.8)
        rows[1]["metrics"]["faithfulness"] = evaluation.metric_result(status="failed")
        rows[2]["metrics"]["faithfulness"] = evaluation.metric_result(status="skipped")
        self.assertEqual(evaluation.aggregate(rows)["hybrid"]["faithfulness"], {"mean": 0.8, "valid": 1, "total": 3, "excluded": 2})

    def test_compare_uses_only_paired_successes(self):
        rows = []
        for case_id, before, after in [("a", 0.5, 0.75), ("b", 1.0, None)]:
            for variant, score in [("hybrid", before), ("hybrid_rerank", after)]:
                row = evaluation.new_row({"id": case_id, "question": "q", "reference": None}, variant)
                row["metrics"]["faithfulness"] = evaluation.metric_result(score)
                rows.append(row)
        self.assertEqual(evaluation.paired_deltas(rows)["faithfulness"], {"mean_delta": 0.25, "paired_valid": 1})

    def test_raw_errors_do_not_leak_keys(self):
        error = ValueError("Authorization: Bearer secret-key-from-upstream")
        self.assertEqual(evaluation.safe_error(error), "ValueError")

    def test_no_reference_or_context_skips_inapplicable_metrics(self):
        row = evaluation.new_row({"id": "out", "question": "q", "reference": None}, "hybrid")
        row["answer"] = "一般答复"
        metrics = {name: FakeMetric(0.5) for name in evaluation.RAGAS_METRICS}
        ragas_stub = types.SimpleNamespace(SingleTurnSample=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {"ragas": ragas_stub}):
            asyncio.run(evaluation.score_row(row, metrics, types.SimpleNamespace(concurrency=1, timeout=10)))
        self.assertEqual(metrics["answer_relevancy"].calls, 1)
        self.assertEqual(metrics["context_recall"].calls, 0)
        self.assertEqual(row["metrics"]["faithfulness"]["status"], "skipped")

    def test_metric_failure_preserves_success_and_context(self):
        row = evaluation.new_row({"id": "a", "question": "q", "reference": "r"}, "hybrid")
        row.update(answer="a", contexts=["context"])
        metrics = {name: FakeMetric(0.7) for name in evaluation.RAGAS_METRICS}
        metrics["faithfulness"] = FakeMetric(TimeoutError("secret"))
        with patch.dict(sys.modules, {"ragas": types.SimpleNamespace(SingleTurnSample=lambda **kwargs: kwargs)}):
            asyncio.run(evaluation.score_row(row, metrics, types.SimpleNamespace(concurrency=1, timeout=10)))
        self.assertEqual(row["status"], "partial_failure")
        self.assertEqual(row["metrics"]["answer_relevancy"]["value"], 0.7)
        self.assertEqual(row["contexts"], ["context"])

    def test_validation_mode_never_loads_engine_or_judge(self):
        path = self.cases_file([{"id": "a", "question": "q"}])
        args = evaluation.parse_ragas_args(["--cases", str(path), "--project-root", str(self.root), "--output", str(self.root / "report")])
        with patch.object(evaluation, "build_metrics", side_effect=AssertionError("不能调用裁判")):
            self.assertEqual(evaluation.run_ragas(args), 0)
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "validated_only")
        self.assertEqual(report["results"], [])
        self.assertEqual(report["summary"], {})

    def test_default_validation_output_does_not_replace_generation_report(self):
        path = self.cases_file([{"id": "a", "question": "q"}])
        generation = self.root / "outputs/ragas/generation_report.json"
        generation.parent.mkdir(parents=True)
        generation.write_text('{"status":"completed","sentinel":true}', encoding="utf-8")
        with patch.object(evaluation, "BASE", self.root):
            args = evaluation.parse_ragas_args(
                ["--cases", str(path), "--project-root", str(self.root)])
        self.assertEqual(args.output, self.root / "outputs/ragas/input_validation")
        self.assertEqual(evaluation.run_ragas(args), 0)
        self.assertTrue(json.loads(generation.read_text(encoding="utf-8"))["sentinel"])
        validation = json.loads(
            (self.root / "outputs/ragas/input_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(validation["status"], "validated_only")

    def test_run_and_explicit_output_selection(self):
        with patch.object(evaluation, "BASE", self.root):
            self.assertEqual(evaluation.parse_ragas_args(["--run"]).output,
                             self.root / "outputs/ragas/generation_report")
            chosen = self.root / "chosen"
            self.assertEqual(evaluation.parse_ragas_args(["--output", str(chosen)]).output, chosen)

    def test_retrieval_setup_failure_preserves_success_report_and_redacts_error(self):
        path = self.cases_file(
            [{"id": "a", "question": "q", "relevant_chunk_indexes": [1]}])
        output = self.root / "retrieval_report"
        output.with_suffix(".json").write_text(
            '{"status":"completed","sentinel":true}', encoding="utf-8")
        args = types.SimpleNamespace(cases=path, compare_rerank=False, rerank=False, top_k=4,
                                     project_root=self.root, output=output)
        secret = "Authorization: Bearer should-not-leak"
        with patch.object(evaluation, "build_retrieval_engine", side_effect=ValueError(secret)):
            self.assertEqual(evaluation.run_retrieval(args), 1)
        self.assertTrue(json.loads(output.with_suffix(".json").read_text(
            encoding="utf-8"))["sentinel"])
        failed = (self.root / "retrieval_report_failed.json").read_text(encoding="utf-8")
        self.assertNotIn(secret, failed)
        self.assertIn("ValueError", failed)

    def test_finish_script_requires_explicit_paid_gate(self):
        script = (SCRIPT.parents[1] / "deploy/local/finish_rag.sh").read_text(encoding="utf-8")
        self.assertIn('--run-paid) RUN_PAID=true', script)
        guard = script.index('if [[ "$RUN_PAID" == true ]]')
        paid_call = script.index('evaluate.sh --run --limit 2 --compare-rerank')
        self.assertGreater(paid_call, guard)

    def test_main_dispatches_subcommands_without_running_them(self):
        with patch.object(evaluation, "retrieval_main", return_value=7) as retrieval:
            self.assertEqual(evaluation.main(["retrieval", "--top-k", "3"]), 7)
            retrieval.assert_called_once_with(["--top-k", "3"])
        with patch.object(evaluation, "ragas_main", return_value=8) as ragas:
            self.assertEqual(evaluation.main(["ragas", "--limit", "1"]), 8)
            ragas.assert_called_once_with(["--limit", "1"])

    def test_live_flow_uses_no_memory_and_records_each_variant(self):
        path = self.cases_file([{"id": "a", "question": "q"}])
        args = evaluation.parse_ragas_args(["--run", "--compare-rerank", "--cases", str(path), "--project-root", str(self.root), "--output", str(self.root / "report")])
        calls = []
        class Engine:
            def __init__(self, with_memory):
                self.assert_no_memory = with_memory is False
                calls.append(("with_memory", with_memory))
                self.reranker, self.model, self.llm = object(), "test-model", types.SimpleNamespace(base_url="https://example.invalid/v1")
                self.options = {}
            def retrieve(self, question, top_k, use_rerank):
                snapshot = json.loads(args.output.with_suffix(".json").read_text(encoding="utf-8"))
                calls.append(("snapshot", snapshot["status"],
                              [row["status"] for row in snapshot["results"]]))
                calls.append(("retrieve", use_rerank))
                return [{"entity": {"text": "from retrieval", "source": "guide", "chunk_index": 1}}]
            def generate(self, question, history, hits):
                calls.append(("generate", history))
                return "generated answer"
        metrics = {name: FakeMetric(0.7) for name in evaluation.RAGAS_METRICS}
        modules = {"app.single_app": types.SimpleNamespace(RAG=Engine), "ragas": types.SimpleNamespace(SingleTurnSample=lambda **kwargs: kwargs)}
        with patch.dict(sys.modules, modules), patch.object(evaluation, "package_versions", return_value={"ragas": "0.2.15"}), patch.object(evaluation, "build_metrics", return_value=metrics):
            self.assertEqual(evaluation.run_ragas(args), 0)
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["results"]), 2)
        self.assertIn(("with_memory", False), calls)
        self.assertIn(("retrieve", False), calls)
        self.assertIn(("retrieve", True), calls)
        self.assertEqual(report["results"][0]["contexts"], ["from retrieval"])
        snapshots = [call for call in calls if call[0] == "snapshot"]
        self.assertEqual(snapshots[0], ("snapshot", "running", ["pending", "pending"]))


if __name__ == "__main__":
    unittest.main()
