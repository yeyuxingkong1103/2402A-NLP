"""Query 改写/扩写单元测试"""
from src.rag.query_rewriter import QueryRewriter


def test_rewrite_no_history():
    rw = QueryRewriter(llm_client=None)
    assert rw.rewrite("同仁堂什么时候成立的？") == "同仁堂什么时候成立的？"


def test_rewrite_anaphora_with_history():
    rw = QueryRewriter(llm_client=None)
    history = [{"role": "user", "content": "同仁堂前身是什么？"}]
    rewritten = rw.rewrite("它是什么时候创立的？", history)
    assert "同仁堂前身是什么？" in rewritten


def test_expand_without_llm():
    rw = QueryRewriter(llm_client=None)
    out = rw.expand("请问一下变压器绝缘油的检测方法是什么？")
    assert out and out[0] == "请问一下变压器绝缘油的检测方法是什么？"
    assert isinstance(out, list)


def test_rewrite_disabled():
    rw = QueryRewriter(llm_client=None)
    rw.enabled = False
    history = [{"role": "user", "content": "同仁堂前身是什么？"}]
    assert rw.rewrite("它是什么时候创立的？", history) == "它是什么时候创立的？"
