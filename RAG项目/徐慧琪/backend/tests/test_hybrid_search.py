# 混合检索测试。filter 表达式与参数是纯函数，可离线断言；
# 真正的双路召回需要 Milvus 在线且有数据。
import pytest

from app.db.milvus import (
    CHILD_TYPES, DENSE_TOP_K, FILTER_EXPR, RRF_K, SPARSE_TOP_K,
    build_filter_expr, entity_count, get_client, hybrid_search,
)


def test_rrf_k_is_60_per_spec():
    # 技术方案 5.2 定的 RRF k=60，改它等于改排序行为
    assert RRF_K == 60


def test_top_k_match_spec():
    assert DENSE_TOP_K == 50
    assert SPARSE_TOP_K == 50


def test_filter_expr_excludes_fathers():
    # 父块有向量但语义太宽，会挤掉精确的子块——这是"3388 全编码"的配套措施
    expr = build_filter_expr("现行有效", child_only=True)
    assert "chunk_type" in expr
    assert "father" not in expr
    assert set(CHILD_TYPES) == {"paragraph", "item"}


def test_filter_expr_embeds_status():
    # 效力过滤是 FR-3.3 的硬要求：废止的法条绝不能被召回
    assert 'status == "现行有效"' in build_filter_expr("现行有效")


def test_filter_expr_without_child_only_omits_chunk_type():
    # 需要按条取整条时（将来的父块直接检索）要能放开
    assert "chunk_type" not in build_filter_expr("现行有效", child_only=False)


def test_default_filter_expr_matches_builder():
    # 两条路径不能各写一遍常量，否则改一处漏一处
    assert FILTER_EXPR == build_filter_expr("现行有效", child_only=True)


def _milvus_available() -> bool:
    try:
        get_client().list_collections()
        return True
    except Exception:
        return False


def _row_count() -> int:
    # 集合可能还不存在（Task 5 没跑），此时 entity_count 会抛错；
    # 收集阶段抛错会让整个文件报 error 而不是 skip，故在这里兜住。
    # 用 entity_count 而非 get_collection_stats：后者是累计计数，
    # 重跑灌库后会虚高，这个 skipif 就永远不成立
    try:
        return entity_count(get_client())
    except Exception:
        return 0


requires_milvus = pytest.mark.skipif(not _milvus_available(), reason="Milvus 未在线")
requires_data = pytest.mark.skipif(
    not _milvus_available() or _row_count() < 3388,
    reason="集合未灌满（需先跑 Task 5 Step 5）",
)


@requires_data
def test_hybrid_search_returns_child_chunks_only():
    # 检索结果里出现 father 就是过滤失效
    client = get_client()
    # 用集合里真实存在的一条文本重新编码，保证查询向量在语义上确实有邻居
    from app.ingest.embed import encode_texts, load_model
    model = load_model()
    dense, sparse = encode_texts(model, ["买卖合同标的物质量不符合约定的违约责任"])[0]
    hits = hybrid_search(client, dense, sparse, top_k=10)
    assert hits, "混合检索没有返回任何结果"
    assert all(h["chunk_type"] in CHILD_TYPES for h in hits)


@requires_data
def test_hybrid_search_returns_expected_fields():
    from app.ingest.embed import encode_texts, load_model
    model = load_model()
    dense, sparse = encode_texts(model, ["合同的违约责任"])[0]
    hits = hybrid_search(get_client(), dense, sparse, top_k=3)
    # top_k 是调用方唯一的条数闸门；limit 写死成 10 也不会报错，
    # 只会让下游 prompt 预算悄悄失效——而本项目无版本控制，事后无可回溯
    assert len(hits) == 3
    first = hits[0]
    for name in ("chunk_id", "article_no", "path", "text", "chunk_type"):
        assert name in first, f"缺少字段 {name}"


@requires_milvus
def test_hybrid_search_on_empty_collection_returns_empty():
    # 空集合应当返回空列表而不是抛错——上层据此判断"没找到依据"，
    # 而不是把异常包装成系统错误
    client = get_client()
    if not client.has_collection("law_chunks"):
        pytest.skip("集合不存在")
    hits = hybrid_search(client, [0.0] * 1023 + [1.0], {1: 0.5},
                         top_k=3, status="绝无此效力状态")
    assert hits == []
