# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
# 工单编号：人工智能NLP-RAG-混合检索任务
# -*- coding: utf-8 -*-
"""benchmark_load.py 冒烟测试：极小规模跑通分档流程与统计字段。

不加载任何真实模型、不建索引、不联网：
  - ``src.rag.ask`` 被 monkeypatch 成假函数（一次成功、一次抛异常），
    因此不触发 embedding/reranker/LLM；
  - GPU 探测被 monkeypatch 为不可用，避免 import torch 与真实显存查询；
  - 预热走 ``--no-warmup`` 等价路径（不调用 rag.warmup）。

只断言压测**流程与统计口径**：并发档位、请求数、成功/失败、错误率、
P50 等字段齐全，以及每题明细落盘。
"""
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_load  # noqa: E402


def test_benchmark_smoke_two_concurrency(tmp_path, monkeypatch):
    from src import rag

    # 假 ask：第 1 次抛异常、第 2 次成功 —— 固定覆盖成功/失败两条记账路径
    lock = threading.Lock()
    state = {"calls": 0}

    def fake_ask(question, use_cache=True, **kwargs):
        with lock:
            state["calls"] += 1
            nth = state["calls"]
        if nth == 1:
            raise RuntimeError("fake failure for smoke test")
        return rag.RagResult(question=question, answer="假答案",
                             mode="rag", timings={"total_ms": 1.0})

    monkeypatch.setattr(rag, "ask", fake_ask)
    # 避免 import torch / 查询 CUDA：显存字段应如实为「未测得」
    monkeypatch.setattr(benchmark_load, "_cuda_available", lambda: False)

    out = tmp_path / "loadtest_result.json"
    rc = benchmark_load.main(["--levels", "2", "--per-level", "2",
                              "--no-warmup", "--out", str(out)])
    assert rc == 0
    assert state["calls"] == 2
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["workorder"] == "人工智能NLP-RAG-混合检索任务"
    assert payload["os"]["python"]                      # os 信息齐全
    assert payload["warmup"]["enabled"] is False

    assert len(payload["levels"]) == 1
    lv = payload["levels"][0]
    assert lv["concurrency"] == 2
    assert lv["n_requests"] == 2                        # 请求数
    assert lv["ok"] == 1 and lv["errors"] == 1          # 成功/失败
    assert lv["error_rate"] == 0.5                      # 错误率
    assert lv["error_kinds"] == {"RuntimeError": 1}     # 错误类型样例
    for key in ("p50_ms", "p90_ms", "p95_ms", "mean_ms", "max_ms"):
        assert isinstance(lv[key], float) and lv[key] >= 0.0
    assert 0.0 <= lv["within_3s_rate"] <= 1.0
    assert lv["qps"] >= 0.0
    assert len(lv["requests"]) == 2                     # 每题明细
    assert lv["gpu_peak_mb"] is None                    # 未测得，不编造数字
    assert "未测得" in (lv["gpu_note"] or "")
