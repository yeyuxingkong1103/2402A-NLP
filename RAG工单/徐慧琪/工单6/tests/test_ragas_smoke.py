# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
模块：evaluate_ragas.py 的**离线冒烟测试**（不加载模型、不调用 Ollama）

只覆盖纯逻辑：参数解析、NaN→None 的失败记录、失败题归并、路径解析、
裁判模型缺失/连接失败时的报错口径。真实评估仍用：
    python evaluate_ragas.py --limit 2 --out data/eval/ragas_smoke.json
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate_ragas as ev


def test_parse_metrics_default_order():
    assert ev._parse_metrics(ev._DEFAULT_METRICS) == list(ev._ALL_METRICS)


def test_parse_metrics_rejects_unknown():
    with pytest.raises(ValueError):
        ev._parse_metrics("faithfulness,not_a_metric")


def test_resolve_path_is_relative_to_project_root():
    p = ev._resolve_path(os.path.join("data", "eval", "x.json"))
    assert os.path.isabs(p)
    assert p.endswith(os.path.join("data", "eval", "x.json"))


def _row(qid: int = 1) -> dict:
    return {"id": qid, "question": "q", "answer": "a", "mode": "rag",
            "contexts": ["c1", "c2"]}


def test_scores_round_and_nan_to_none():
    df = pd.DataFrame({"faithfulness": [0.66666], "context_recall": [float("nan")]})
    errors: list = []
    per_q = ev._scores_from_df(df, [_row()], ["faithfulness", "context_recall"], errors)
    assert per_q[0]["faithfulness"] == 0.667          # 保留三位小数
    assert per_q[0]["context_recall"] is None         # NaN → None，不插补
    assert errors and errors[0]["metric"] == "context_recall" and errors[0]["id"] == 1


def test_scores_missing_column_recorded():
    df = pd.DataFrame({"faithfulness": [1.0]})
    errors: list = []
    per_q = ev._scores_from_df(df, [_row(7)], ["faithfulness", "context_precision"], errors)
    assert per_q[0]["context_precision"] is None
    assert any(e.get("metric") == "context_precision" and e.get("id") == 7 for e in errors)


def test_failed_questions_aggregate_by_id():
    errors = [
        {"stage": "ragas", "id": 3, "metric": "faithfulness", "error": "NaN"},
        {"stage": "ask", "id": 3, "error": "boom"},
        {"stage": "ragas", "id": 4, "metric": "context_recall", "error": "NaN"},
    ]
    out = ev._failed_questions(errors)
    assert [r["id"] for r in out] == ["3", "4"]
    assert len(out[0]["reasons"]) == 2                # 同一题的多条原因归并
    assert out[0]["reasons"][0].startswith("[faithfulness]")


def test_check_judge_missing_llm(monkeypatch):
    from src import config, llm
    monkeypatch.setattr(llm, "list_models", lambda force=False: ["bge-m3:latest"])
    with pytest.raises(RuntimeError) as excinfo:
        ev._check_judge()
    assert config.OLLAMA_MODEL in str(excinfo.value)  # 报错点名缺失的裁判模型


def test_check_judge_ollama_down(monkeypatch):
    from src import llm

    def _boom(force=False):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(llm, "list_models", _boom)
    with pytest.raises(RuntimeError) as excinfo:
        ev._check_judge()
    assert "Ollama" in str(excinfo.value)
