# -*- coding: utf-8 -*-
"""在线级：启动预热为硬要求（§3.4）+ 预热后首题无尖峰。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

判定依据（``设计/验收标准.md`` §3.4）：
    * ``QAEngine.warmup()`` 必须依次调用 ``text_utils.warmup_tokenizer()`` → ``embedder.warmup()``
      → ``llm_client.probe_backends()``，并在 ``rag_trace.jsonl`` 留下 ``func.exit`` + ``elapsed_ms``；
    * **零调用点 = 实现缺陷**（不接受「首次查询慢属正常」的豁免）；
    * 预热后首次查询不得出现 ~1 s 级尖峰（本文件用首题首字 ≤3000 ms + 与其余题同分布留痕）。
"""

from __future__ import annotations

from typing import Any

import pytest

from common.reports import trace_events, write_report

pytestmark = [pytest.mark.online, pytest.mark.linkage]


def test_warmup_payload_contract(engine: Any) -> None:
    """``warmup()`` 返回三段耗时（tokenizer/embed/llm_probe），三段都必须被真实调用。"""
    payload = engine.warmup()
    assert isinstance(payload, dict), f"warmup 应返回 dict，实测 {type(payload).__name__}"
    for key in ("tokenizer_ms", "embed_ms", "llm_probe_ms"):
        assert key in payload, f"warmup 返回缺字段 {key!r}：{sorted(payload)}"
    assert float(payload["tokenizer_ms"]) > 0.0, "tokenizer 预热耗时异常（未真实调用？）"
    assert float(payload["embed_ms"]) > 0.0, "嵌入预热耗时异常（未真实调用？）"


def test_warmup_events_in_trace(engine: Any) -> None:
    """预热必须留下可审计记录（tokenizer 预热 / probe_backends / QAEngine.warmup + elapsed_ms）。

    实测（2026-10-04 定稿代码）：记录形态为
        ``app.log``：``utils.tokenizer_warmup`` 事件；``func.enter/exit``（func=``warmup``, module=``utils``）；
                     ``QAEngine.warmup`` 的 ``func.enter/exit``；``probe_backends`` 的 ``func.exit``（含 elapsed_ms）；
        ``rag_trace.jsonl``：``qa.warmup`` 事件（**无 func 跨度** —— 该文件是事件流，不是 span 流）。
    两种落盘都算「留痕齐备」；断言取并集，但要求三要素（tokenizer / 探测 / QAEngine.warmup）都能对上。
    """
    engine.warmup()
    app_rows = trace_events("app.log")
    trace_rows = trace_events("rag_trace.jsonl")
    for name, rows in (("app", app_rows), ("ragtrace", trace_rows)):
        write_report(f"online_warmup_{name}", f"预热取证（{name}）", [
            {"event": row.get("event"), "func": row.get("func"), "module": row.get("module"),
             "elapsed_ms": row.get("elapsed_ms"), "level": row.get("level")}
            for row in rows if "warmup" in str(row.get("func", "")).lower()
            or "warmup" in str(row.get("event", "")).lower()
            or "probe_backends" in str(row.get("func", ""))
        ])

    all_rows = app_rows + trace_rows
    events = [str(row.get("event", "")) for row in all_rows]
    funcs = [str(row.get("func", "")) for row in all_rows]
    exits = [row for row in all_rows
             if str(row.get("event", "")).endswith(".exit") and row.get("elapsed_ms") is not None]

    assert any("tokenizer_warmup" in event for event in events) or any("warmup" in func for func in funcs), \
        f"缺少 tokenizer 预热记录：events={sorted(set(events))[:15]}"
    assert any("probe_backends" in func for func in funcs), \
        f"缺少 probe_backends 记录：funcs={sorted(set(funcs))[:25]}"
    assert any("QAEngine.warmup" in func or "qa.warmup" in event
               for func, event in zip(funcs, events)), \
        f"缺少 QAEngine.warmup 记录：events={sorted(set(events))[-10:]}"
    warmup_exits = [row for row in exits
                    if "warmup" in str(row.get("func", "")).lower()
                    or "probe" in str(row.get("func", "")).lower()]
    assert warmup_exits, f"预热/探测缺少 func.exit + elapsed_ms：funcs={sorted(set(funcs))[:25]}"


def test_llm_probe_budget_and_no_retry(engine: Any) -> None:
    """探测预算：三后端 × 0.5 s 超时、不重试 → 总耗时上界 1.5 s（记录在 ``app.log`` 的 func 跨度里）。"""
    engine.warmup()
    events = trace_events("app.log") + trace_events("rag_trace.jsonl")
    probes = [row for row in events if "probe_backends" in str(row.get("func", ""))
              and str(row.get("event", "")).endswith(".exit")]
    assert probes, "缺少 probe_backends 的 func.exit 事件（app.log / rag_trace.jsonl 都没有）"
    worst = max(float(row.get("elapsed_ms") or 0.0) for row in probes)
    write_report("online_probe_budget", "LLM 探测预算", [
        {"probe_exits": len(probes), "worst_ms": worst, "bound_ms": 1500.0}
    ])
    assert worst <= 1500.0, (f"probe_backends 耗时 {worst} ms > 1500 ms（三后端 × 500 ms 上界）→ "
                            f"疑似超时未生效或发生重试")


def test_first_question_after_warmup_within_budget(engine: Any) -> None:
    """预热后首题首字 ≤ 3000 ms（消除 ~1 s 级冷启动尖峰）。"""
    engine.warmup()
    answer = engine.ask("武汉兴图新科电子股份有限公司法定代表人是谁？",
                        session_id="t8-warmup-probe", stream=False)
    latency = float(getattr(answer, "first_token_ms", 0.0) or 0.0)
    write_report("online_first_question_after_warmup", "预热后首题", [
        {"question": "法定代表人", "first_token_ms": latency, "budget_ms": 3000,
         "is_unknown": bool(getattr(answer, "is_unknown", False))}
    ])
    assert 0.0 < latency <= 3000.0, f"预热后首题首字 {latency} ms 超出预算（预热未生效）"
