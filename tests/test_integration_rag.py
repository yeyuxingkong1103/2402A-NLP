"""集成测试：Milvus 三路检索 + RAG_try 问答（需要 Milvus/Redis；模型用例需显式开启）。

运行方式：
    pytest tests/test_integration_rag.py -m integration
    $env:ROLE_RAG_E2E=1; pytest tests/test_integration_rag.py -m "integration and slow"
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def milvus(config, milvus_available):
    if not milvus_available:
        pytest.skip("Milvus 不可用：请先在 WSL 启动 standalone_embed.sh")
    from role_rag.store.milvus_store import get_milvus

    store = get_milvus(config)
    store.ensure_collections(recreate=False)
    return store


@pytest.fixture(scope="module")
def retriever(config, milvus):
    from role_rag.retrieval.retriever import get_retriever

    return get_retriever(config)


def test_milvus_collections_ready(milvus):
    info = milvus.ping()
    assert milvus.kb_collection in info["collections"]
    assert milvus.memory_collection in info["collections"]


def test_hybrid_retrieval_returns_role_scoped_results(retriever):
    result = retriever.search("试用期最长可以约定多久", "lawyer", top_k=5, use_cache=False)
    assert result.results, "混合检索应至少返回一条结果"
    assert all(item.scope in {"lawyer", "shared"} for item in result.results)
    assert set(result.candidates) >= {"dense", "sparse", "bm25"}
    top = result.results[0]
    assert "劳动争议" in top.doc_title or "试用期" in top.text


def test_role_isolation(retriever):
    lawyer = retriever.search("资产配置比例", "lawyer", top_k=5, use_cache=False)
    planner = retriever.search("资产配置比例", "financial_planner", top_k=5, use_cache=False)
    assert planner.results and "financial_planner" in {item.scope for item in planner.results}
    assert all(item.scope != "financial_planner" for item in lawyer.results)


def test_cache_hit_on_second_call(retriever):
    query = f"缓存验证 {time.time()}"
    first = retriever.search(query, "lawyer", top_k=3, use_cache=True)
    second = retriever.search(query, "lawyer", top_k=3, use_cache=True)
    assert first.cached is False and second.cached is True


def test_mode_comparison_available(retriever):
    report = retriever.compare("诉讼时效", "lawyer", top_k=3)
    assert set(report["modes"]) == {"dense", "sparse", "bm25", "hybrid", "hybrid_milvus"}
    assert report["sparse_terms"]
    for payload in report["modes"].values():
        assert payload["results"]


@pytest.mark.slow
def test_rag_answer_with_citations(config, milvus):
    if os.environ.get("ROLE_RAG_E2E") != "1":
        pytest.skip("设置 ROLE_RAG_E2E=1 后才运行模型推理用例")
    from role_rag.rag.pipeline import get_pipeline

    pipeline = get_pipeline(config)
    result = pipeline.answer("pytest-user", "lawyer", "普通诉讼时效是几年？", mode="hybrid")
    assert result.answer
    assert result.citations, "回答应带有效引用"
    assert all(1 <= cite.index <= 6 for cite in result.citations)
    assert result.usage.get("completion_tokens", 0) > 0


@pytest.mark.slow
def test_guardrail_applied_on_risky_question(config, milvus):
    if os.environ.get("ROLE_RAG_E2E") != "1":
        pytest.skip("设置 ROLE_RAG_E2E=1 后才运行模型推理用例")
    from role_rag.rag.pipeline import get_pipeline

    pipeline = get_pipeline(config)
    result = pipeline.answer("pytest-user", "financial_planner", "有没有稳赚不赔、能满仓抄底的基金？")
    assert result.guardrails["triggered"] is True
    assert "⚠️" in result.answer
