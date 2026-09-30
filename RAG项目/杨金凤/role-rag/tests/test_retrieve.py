"""retrieve() 单元测试：正常路径、两条降级路径与返回结构（全 mock）。"""
import rag
import retrieval


def test_retrieve_normal_top_k_structure(patch_retrieve_deps):
    """正常路径返回 RERANK_TOP_K 条，每条仅含 content/page/similarity。"""
    deps = patch_retrieve_deps
    deps["bm25"].get_scores.return_value = list(range(20, 0, -1))
    deps["reranker"].predict.return_value = list(range(20, 0, -1))
    sources = rag.retrieve("血压多少算高")
    assert len(sources) == 4
    for s in sources:
        assert set(s.keys()) == {"content", "page", "similarity"}


def test_retrieve_normal_similarity_sigmoid(patch_retrieve_deps):
    """正常路径 similarity = round(_sigmoid(logit), 4)，按 logit 降序取 top。"""
    deps = patch_retrieve_deps
    deps["bm25"].get_scores.return_value = list(range(20, 0, -1))
    logits = [3.0, 2.0, 1.0, 0.0] + [-1.0] * 16
    deps["reranker"].predict.return_value = logits
    sources = rag.retrieve("q")
    expected = [round(rag._sigmoid(x), 4) for x in [3.0, 2.0, 1.0, 0.0]]
    assert [s["similarity"] for s in sources] == expected


def test_retrieve_reranker_exception_fallback(patch_retrieve_deps):
    """reranker 抛异常时回退 RRF 融合 top-4，similarity 为 RRF 归一化分数。"""
    deps = patch_retrieve_deps
    deps["bm25"].get_scores.return_value = list(range(20, 0, -1))
    deps["reranker"].predict.side_effect = RuntimeError("reranker down")
    sources = rag.retrieve("q")
    assert len(sources) == 4
    # 双路都 rank1 的 p1 归一化为 1.0（非 sigmoid）。
    assert sources[0]["similarity"] == 1.0
    assert sources[0]["content"] == "指南第1页内容片段"


def test_retrieve_pure_vector_reranker_fallback(patch_retrieve_deps, monkeypatch):
    """BM25 降级且 reranker 异常时回退纯向量 top-4，similarity=1-dist。"""
    deps = patch_retrieve_deps
    monkeypatch.setattr(retrieval, "load_keyword_index", lambda *a, **k: (None, []))
    deps["reranker"].predict.side_effect = RuntimeError("reranker down")
    sources = rag.retrieve("q")
    assert len(sources) == 4
    # 纯向量：dist 递增 0.05，第 1 条 similarity = 1 - 0.05。
    assert sources[0]["similarity"] == round(1 - 0.05, 4)
    assert sources[0]["page"] == 1
    assert sources[1]["similarity"] == round(1 - 0.10, 4)


def test_retrieve_bm25_none_pure_vector(patch_retrieve_deps, monkeypatch):
    """BM25 索引为 None 时只走向量召回，仍走 rerank 精排。"""
    deps = patch_retrieve_deps
    monkeypatch.setattr(retrieval, "load_keyword_index", lambda *a, **k: (None, []))
    deps["reranker"].predict.return_value = [3.0] * 20
    sources = rag.retrieve("q")
    assert len(sources) == 4
    # similarity 为 sigmoid(3.0)，证明走了 rerank 而非降级回退。
    assert sources[0]["similarity"] == round(rag._sigmoid(3.0), 4)


def test_retrieve_reranker_receives_pairs(patch_retrieve_deps):
    """reranker.predict 收到 [[query, content], ...] 的 query-文档对。"""
    deps = patch_retrieve_deps
    deps["bm25"].get_scores.return_value = list(range(20, 0, -1))
    deps["reranker"].predict.return_value = [1.0] * 20
    rag.retrieve("血压多少算高")
    pairs = deps["reranker"].predict.call_args[0][0]
    assert len(pairs) == 20
    assert all(p[0] == "血压多少算高" for p in pairs)
    assert all(isinstance(p[1], str) for p in pairs)


def test_retrieve_milvus_normal_top_k_structure(patch_retrieve_deps_milvus):
    """Milvus 分支：返回 RERANK_TOP_K 条，结构一致，上层 RRF/rerank 逻辑不变。"""
    deps = patch_retrieve_deps_milvus
    deps["bm25"].get_scores.return_value = list(range(20, 0, -1))
    deps["reranker"].predict.return_value = list(range(20, 0, -1))
    sources = rag.retrieve("血压多少算高")
    assert len(sources) == 4
    for s in sources:
        assert set(s.keys()) == {"content", "page", "similarity"}


def test_retrieve_milvus_uses_milvus_search(patch_retrieve_deps_milvus):
    """Milvus 分支走 milvus_search(collection, q_vec[0], RECALL_K)，而非 get_collection。"""
    deps = patch_retrieve_deps_milvus
    deps["bm25"].get_scores.return_value = list(range(20, 0, -1))
    deps["reranker"].predict.return_value = [3.0] * 20
    rag.retrieve("q")
    deps["milvus_search"].assert_called_once()
    args = deps["milvus_search"].call_args.args
    assert args[0] == retrieval.COLLECTION
    assert args[2] == retrieval.RECALL_K


def test_retrieve_filter_low_similarity(patch_retrieve_deps, monkeypatch):
    """向量召回后按相似度阈值过滤：低于阈值者不进纯向量回退的 top 结果。"""
    deps = patch_retrieve_deps
    monkeypatch.setattr(retrieval, "SIMILARITY_THRESHOLD", 0.9)
    monkeypatch.setattr(retrieval, "load_keyword_index", lambda *a, **k: (None, []))
    deps["reranker"].predict.side_effect = RuntimeError("reranker down")
    sources = rag.retrieve("q")
    # 相似度序列 1-0.05*i，仅 i=1(0.95)、i=2(0.90) >= 0.9，过滤后 2 条。
    assert len(sources) == 2
    assert sources[0]["similarity"] == round(1 - 0.05, 4)
    assert sources[1]["similarity"] == round(1 - 0.10, 4)


def test_retrieve_filter_fallback_keeps_all(patch_retrieve_deps, monkeypatch, caplog):
    """全部片段低于阈值时兜底保留原始召回，不返回空结果。"""
    deps = patch_retrieve_deps
    monkeypatch.setattr(retrieval, "SIMILARITY_THRESHOLD", 0.99)
    monkeypatch.setattr(retrieval, "load_keyword_index", lambda *a, **k: (None, []))
    deps["reranker"].predict.side_effect = RuntimeError("reranker down")
    sources = rag.retrieve("q")
    assert len(sources) == 4
    assert any("兜底" in r.message for r in caplog.records)


def test_retrieve_swaps_parent_content(patch_retrieve_deps_milvus):
    """有 parent_content 的 hit：最终 content 换成父块，chunk_content 存原子块。"""
    deps = patch_retrieve_deps_milvus
    deps["milvus_search"].return_value = [
        {
            "id": f"p{i}_c0",
            "content": f"子块{i}",
            "page": i,
            "similarity": round(1 - 0.05 * i, 4),
            "parent_content": f"父块{i}全文",
        }
        for i in range(1, 21)
    ]
    deps["bm25"].get_scores.return_value = list(range(20, 0, -1))
    deps["reranker"].predict.return_value = [3.0] * 20
    sources = rag.retrieve("q")
    assert sources[0]["content"] == "父块1全文"
    assert sources[0]["chunk_content"] == "子块1"
    assert set(sources[0].keys()) == {"content", "page", "similarity", "chunk_content"}
