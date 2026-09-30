# -*- coding: utf-8 -*-
"""检索过滤的回归测试。

本模块在守护什么
================

2026-09-16 发现并修复了一个**长期存在、且完全静默**的 bug：

    `retrieval._qdrant_hybrid` 把 `query_filter` 放在 `query_points` 的**顶层**，
    但该查询用的是 `prefetch + FusionQuery(RRF)`。**顶层 filter 对融合不起作用**
    —— 融合只合并两路的排名，不重新过滤。

    后果：`检索调试台` 传任何过滤条件都被忽略，查询结果与不过滤完全一致，
    且两边的返回都是 HTTP 200，没有任何报错。

    实测对照（传入一个绝不存在的过滤值）：
        顶层 filter + fusion        -> 仍返回 5 条（被忽略）
        每个 Prefetch 挂 filter      -> 正确返回 0 条

因此这里用「**传入不存在的值必须返回 0 条**」来锁死该行为 ——
这个断言简单、直接，且正是当初被违反的性质。仅断言「传对的值能返回结果」
是不够的：过滤被忽略时，传对的值同样会返回结果。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


@pytest.fixture()
def qdrant_store(monkeypatch):
    """把向量库切到 qdrant（默认后端）。"""
    from app.core import config
    monkeypatch.setattr(config, "VECTOR_STORE", "qdrant")
    return config


def test_filter_excludes_nonexistent_value(qdrant_store, isolated_qdrant):
    """核心回归：过滤一个绝不存在的值，必须返回 0 条。

    修复前这条会失败（返回满额结果），因为 filter 被融合查询忽略了。
    """
    from app.services import retrieval

    hits, _ = retrieval.hybrid_search(
        "试用期最长多久", "kb_legal", 5,
        filters={"law_name": "绝不存在的法名xyzabc"},
    )
    assert hits == [], (
        f"过滤不存在的 law_name 却返回 {len(hits)} 条 —— "
        "说明 filter 没有生效（很可能是又回到了「顶层 query_filter + fusion」的写法）"
    )


def test_filter_by_existing_source_works(qdrant_store, isolated_qdrant):
    """传存在的值应能命中，证明过滤不是「一律返回空」。"""
    from qdrant_client import models as qm

    from app.services.qdrant_store import get_client
    from app.services import retrieval

    client = get_client()
    points, _ = client.scroll(collection_name="kb_legal", limit=1, with_payload=True)
    assert points, "kb_legal 为空，无法测试"
    source = (points[0].payload or {}).get("source")
    assert source, "payload 缺 source 字段"

    hits, _ = retrieval.hybrid_search(
        "试用期最长多久", "kb_legal", 5, filters={"source": source},
    )
    assert hits, f"按存在的 source={source} 过滤却返回 0 条"
    assert all(h.get("source") == source for h in hits), (
        "返回结果里混入了非目标 source —— 过滤未真正约束结果集"
    )


def test_no_filter_returns_results(qdrant_store, isolated_qdrant):
    """不传 filters 时应正常返回（确认上面的空结果不是别的原因导致的）。"""
    from app.services import retrieval

    hits, _ = retrieval.hybrid_search("试用期最长多久", "kb_legal", 5)
    assert len(hits) > 0, "不过滤时返回 0 条 —— 检索本身有问题"


def test_empty_filter_dict_is_treated_as_no_filter(qdrant_store, isolated_qdrant):
    """空 dict / 全空值的 filters 应等同不过滤，而不是返回 0 条。"""
    from app.services import retrieval

    base, _ = retrieval.hybrid_search("试用期最长多久", "kb_legal", 5)
    for empty in ({}, {"law_name": None}, {"law_name": ""}):
        hits, _ = retrieval.hybrid_search("试用期最长多久", "kb_legal", 5, filters=empty)
        assert len(hits) == len(base), f"filters={empty} 被当成了有效过滤条件"
