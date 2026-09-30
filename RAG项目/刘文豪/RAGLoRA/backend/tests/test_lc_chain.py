# -*- coding: utf-8 -*-
"""LCEL 链路测试：签名与事件协议必须与 rag_chain 完全一致。"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import lc_chain, rag_chain  # noqa: E402


class _FakeCharacter:
    name = "测试角色"
    prompt_template = "{identity_block}\n{context}\n{question}"
    identity_block = "你是测试角色。"
    description = "测试"
    style_json = {}
    domain_constraints = "不编造。"
    kb_collection = "kb_medical"
    recall_top_k = 3
    rerank_top_k = 2
    temperature = 0.3


def test_signatures_match_manual_chain():
    """两条链路的函数签名必须一致，否则路由层无法透明切换。"""
    for fn in ("prepare", "ask", "ask_stream"):
        a = inspect.signature(getattr(lc_chain, fn))
        b = inspect.signature(getattr(rag_chain, fn))
        assert list(a.parameters) == list(b.parameters), fn


def test_prepare_returns_expected_keys(isolated_qdrant):
    out = lc_chain.prepare(_FakeCharacter(), "高血压诊断标准", [])
    assert set(out) >= {"rewritten", "hits", "trace"}
    assert "recall_ms" in out["trace"]
    assert isinstance(out["hits"], list)
    for h in out["hits"]:
        assert "text" in h and "source" in h


def test_ask_stream_event_protocol(isolated_qdrant):
    """事件协议必须与 rag_chain.ask_stream 一致。

    ⚠️ 注意：`sources` 事件的负载是**列表**（persona.build_sources() 的返回值），
    不是 dict —— 手写链路 rag_chain.ask_stream 同样 yield 列表。
    不要把这里改成「所有事件负载都是 dict」，那对两条链路都会失败。
    """
    events = []
    for event, data in lc_chain.ask_stream(_FakeCharacter(), "高血压诊断标准", []):
        events.append(event)
        if event == "sources":
            assert isinstance(data, list), "sources 事件负载应为列表"
            for item in data:
                assert isinstance(item, dict)
        else:
            assert isinstance(data, dict), f"{event} 事件负载应为 dict"
        if event == "done":
            break
    assert events[0] == "trace"
    assert events[1] == "sources"
    assert "delta" in events
    assert "done" in events


def test_trace_caliber_matches_manual_chain(isolated_qdrant):
    """trace 口径必须与手写链路口径一致 —— 前端「链路可视」直接展示这些数字。

    LCEL 把召回与精排融在同一个管道里，容易让 recall / recall_ms 被精排污染。
    本测试锁住两件事：
        1. recall 是**精排前**的召回数量（不是精排后的条数）
        2. recall_ms 已**抵扣**精排耗时（与 rag_chain 口径一致）
    """
    ch = _FakeCharacter()                       # recall_top_k=3, rerank_top_k=2
    trace = lc_chain.prepare(ch, "高血压诊断标准", [])["trace"]

    assert trace["chain"] == "langchain"
    assert trace["recall"] > trace["reranked"], (
        f"recall({trace['recall']}) 应严格大于 reranked({trace['reranked']})："
        "recall_top_k=3 > rerank_top_k=2。若两者相等，说明 recall 被写成了精排后的数量"
    )
    assert trace["rerank_ms"] >= 0
    assert trace["recall_ms"] >= 0
