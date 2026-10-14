# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
import json

from src.compare_with_wo07 import (fact_recall, load_wo07_results, summarize,
                                   _wo07_seconds)


def test_fact_recall_all_hit():
    assert fact_recall("不良贷款率1.65%，拨备覆盖率183.12%",
                       ["1.65", "183.12"]) == 1.0


def test_fact_recall_partial():
    r = fact_recall("不良贷款率1.65%", ["1.65", "183.12"])
    assert abs(r - 0.5) < 1e-6


def test_fact_recall_normalizes_commas_and_spaces():
    """千分位与空格差异不应算未命中。"""
    assert fact_recall("营业收入 1,168,867 百万元", ["1168867"]) == 1.0


def test_fact_recall_empty_facts():
    assert fact_recall("任意答案", []) == 0.0


def test_load_wo07_results_handles_missing_file():
    assert load_wo07_results("不存在的路径.json") == {}


def test_load_wo07_results_indexes_by_id(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"id": 1, "answer": "甲"}, {"id": 2, "answer": "乙"}]),
                 encoding="utf-8")
    got = load_wo07_results(str(p))
    assert got[1]["answer"] == "甲"
    assert got[2]["answer"] == "乙"


def test_load_wo07_results_handles_rows_wrapper(tmp_path):
    """工单07 实际产物是 {"rows": [...]} 包装结构，必须能读。"""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"strategy": "hybrid", "top_k": 5,
                             "rows": [{"id": 3, "answer": "丙"}]}),
                 encoding="utf-8")
    got = load_wo07_results(str(p))
    assert got[3]["answer"] == "丙"


def test_wo07_seconds_prefers_wall_ms():
    """工单07 的耗时为 answer_wall_ms（毫秒），应换算为秒。"""
    assert abs(_wo07_seconds({"answer_wall_ms": 4606.4}) - 4.606) < 1e-6


def test_summarize_computes_averages():
    rows = [{"id": 1, "recall_08": 1.0, "recall_07": 0.5, "seconds_08": 2.0},
            {"id": 2, "recall_08": 0.0, "recall_07": 0.0, "seconds_08": 4.0}]
    s = summarize(rows)
    assert abs(s["avg_recall_08"] - 0.5) < 1e-6
    assert abs(s["avg_recall_07"] - 0.25) < 1e-6
    assert abs(s["avg_seconds_08"] - 3.0) < 1e-6
    assert s["n_improved"] == 1


def test_summarize_marks_wo07_uncovered():
    rows = [{"id": 11, "recall_08": 1.0, "recall_07": None, "seconds_08": 2.0}]
    s = summarize(rows)
    assert s["n_wo07_uncovered"] == 1
