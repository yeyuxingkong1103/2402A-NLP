# -*- coding: utf-8 -*-
"""T8 留痕与报告写入（三级测试共用）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

纪律：
    * 兄弟模块不要重复实现「写报告」；报告一律落到 ``测试/留痕/``，**不写** ``优化/评估结果/``
      （那是 optimizer 的产物目录，tester 不得覆盖）；
    * 每份报告都必须带工单编号与「RAGAS 未运行（依赖不可用，本机断网）」标注
      （``设计/接口设计.md`` §10.3 报告纪律），严禁出现任何 RAGAS 数值字段。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import assertions, paths


def write_report(name: str, title: str, rows: list[dict[str, Any]],
                 extra: dict[str, Any] | None = None) -> Path:
    """把逐题/逐项结果写进 ``测试/留痕/<name>.json``（父目录自动创建）。"""
    paths.ensure_trace_dir()
    payload = {
        "work_order": assertions.WORK_ORDER,
        "name": name,
        "title": title,
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ragas": assertions.RAGAS_BANNER,
        "extra": extra or {},
        "rows": rows,
    }
    target = paths.TRACE_DIR / f"{name}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_text(name: str, body: str) -> Path:
    """写纯文本留痕（如 simulate_user 的会话记录）。"""
    paths.ensure_trace_dir()
    target = paths.TRACE_DIR / name
    target.write_text(body, encoding="utf-8")
    return target


def trace_events(name: str = "rag_trace.jsonl", limit: int = 20000) -> list[dict[str, Any]]:
    """读结构化日志末尾若干行并解析为字典列表（坏行跳过）。"""
    path = paths.trace_file(name)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def assert_no_ragas_numbers(payload: Any) -> None:
    """断言报告里**没有任何** RAGAS 数值字段（防止伪造指标混入留痕）。"""
    banned = ("ragas_score", "faithfulness", "answer_relevancy", "context_precision",
              "context_recall", "answer_correctness")
    text = json.dumps(payload, ensure_ascii=False)
    found = [token for token in banned if token in text]
    if found:
        raise AssertionError(f"报告出现 RAGAS 指标字段（严禁伪造）：{found}")
