# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""多轮接入：缓存隔离、Turn 回填、unresolved 不猜测。"""
import pytest

from src import dialogue, rag


XINGTU = "武汉兴图新科电子股份有限公司"
LIYUAN = "武汉力源信息技术股份有限公司"

pytestmark = pytest.mark.slow          # 需要模型与索引；`-m "not slow"` 下跳过


def test_cache_key_includes_resolved_question_and_focus():
    """同一句代词问题在不同会话焦点下含义不同，缓存必须隔开。"""
    k1 = rag._multiturn_cache_key("他参与的哪个工程获奖了？", XINGTU)
    k2 = rag._multiturn_cache_key("他参与的哪个工程获奖了？", LIYUAN)
    assert k1 != k2


def test_ask_with_state_returns_rewrite_info():
    st = dialogue.DialogueState()
    r = rag.ask(f"{XINGTU}的法定代表人是谁？", state=st)
    assert r.answer
    assert r.rewrite["mode"] == "passthrough"
    assert r.rewrite["resolved"] == f"{XINGTU}的法定代表人是谁？"


def test_ask_turn_backfills_state():
    st = dialogue.DialogueState()
    r1, t1 = rag.ask_turn(f"{XINGTU}来自军用领域的收入分别是多少？", st)
    assert t1.idx == 1 and t1.focus_company == XINGTU
    assert st.turn_count == 1

    r2, t2 = rag.ask_turn("这个公司的法定代表人是谁？", st)
    assert t2.idx == 2
    assert XINGTU in t2.resolved_question, "代词轮必须被改写成含公司的独立问题"
    assert t2.coref_span and t2.coref_span.get("kind") in ("demonstrative", "zero")
    assert st.turn_count == 2


def test_ask_turn_unresolved_does_not_guess():
    """无历史的代词问题 → 明确提示，绝不检索瞎答。"""
    st = dialogue.DialogueState()
    r, t = rag.ask_turn("他的法定代表人是谁？", st)
    assert t.rewrite_mode == "unresolved"
    assert "哪家公司" in r.answer
    assert r.mode == "clarify"


def test_cache_hit_refreshes_rewrite_info_to_current_call():
    """F1 回归：缓存命中时 rewrite 留痕必须描述**本次**调用，而不是写缓存的那次。

    同一 (改写后问题, 焦点) 可由不同原始问题、不同改写路径产生（显式点名 →
    passthrough，代词 → coref）。命中时沿用缓存里的旧留痕，会让 ask_turn 回填的
    Turn.rewrite_mode / Turn.coref_span 与当轮不符，污染逐轮评测统计。
    """
    st = dialogue.DialogueState()
    q = f"{XINGTU}的法定代表人是谁？"
    stale = rag.RagResult(
        question="那家公司呢？", answer="旧答案（来自缓存）",
        rewrite={"mode": "coref_llm", "resolved": "旧会话里的改写",
                 "span": {"kind": "demonstrative"}, "latency_ms": 111.0})
    rag._cache_put(rag._multiturn_cache_key(q, st.focus_company), stale)
    try:
        r = rag.ask(q, state=st)
    finally:
        rag.clear_cache()

    assert r.from_cache is True                     # 确实走了命中路径
    assert r.answer == "旧答案（来自缓存）"
    assert r.question == q
    assert r.rewrite["mode"] == "passthrough"       # 留痕来自本次调用
    assert r.rewrite["resolved"] == q
    assert r.rewrite["span"] is None


def test_ask_stream_shares_cache_namespace_with_ask():
    """F2 回归：流式路径写入的缓存必须与 ask() 同一命名空间（同一 key 推导）。

    旧实现写裸问题 key，而 ask() 只查多轮 key：该条目任何读者都读不到，
    且流式答过的问题不再暖 ask 的缓存。ask_stream 无多轮 state → 焦点为 None。
    """
    q = f"{XINGTU}的法定代表人是谁？"
    rag.clear_cache()
    try:
        events = list(rag.ask_stream(q))
        final = events[-1]["result"]
        assert final.answer and final.mode in ("rag", "extractive")

        assert rag._cache_get(q) is None, "不得再写旧的裸问题 key"
        hit = rag._cache_get(rag._multiturn_cache_key(q, None))
        assert hit is not None, "流式结果必须落入 ask(state=None) 的缓存 key"
        assert hit.answer == final.answer

        r = rag.ask(q, use_cache=True)              # 流式结果能暖上 ask 的缓存
        assert r.from_cache is True
        assert r.answer == final.answer
        assert r.rewrite["mode"] == "passthrough"   # F1：命中时留痕刷新为本次
    finally:
        rag.clear_cache()
