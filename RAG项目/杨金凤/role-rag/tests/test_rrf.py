"""_rrf_fuse() 单元测试：RRF 融合计算、归一化、RRF_K 参数与来源标签。"""
import rag


def hit(cid, page=1):
    """构造一条召回 hit（含 id/content/page）。"""
    return {"id": cid, "content": f"内容{cid}", "page": page}


def test_rrf_perfect_overlap():
    """两路完美重叠时 top1 双路命中且相似度归一化为 1.0。"""
    vec = [hit("a"), hit("b"), hit("c")]
    bm25 = [hit("a"), hit("b"), hit("c")]
    cands, source = rag._rrf_fuse(vec, bm25, k=60)
    assert [c["content"] for c in cands] == ["内容a", "内容b", "内容c"]
    assert source == "双路命中"
    assert cands[0]["similarity"] == 1.0


def test_rrf_no_overlap():
    """两路完全不重叠时全部候选合并，top1 来源为仅向量。"""
    vec = [hit("a"), hit("b")]
    bm25 = [hit("c"), hit("d")]
    cands, source = rag._rrf_fuse(vec, bm25, k=60)
    assert len(cands) == 4
    assert source == "仅向量"


def test_rrf_partial_overlap():
    """部分重叠时重叠项排在非重叠项之前。"""
    vec = [hit("a"), hit("b"), hit("c")]
    bm25 = [hit("b"), hit("c"), hit("d")]
    cands, source = rag._rrf_fuse(vec, bm25, k=60)
    assert cands[0]["content"] == "内容b"
    assert source == "双路命中"
    assert len(cands) == 4


def test_rrf_score_formula():
    """score = Σ 1/(k+rank) 手算精确，similarity 按 2/(k+1) 归一化。"""
    vec = [hit("a"), hit("b")]
    bm25 = [hit("a")]
    cands, _ = rag._rrf_fuse(vec, bm25, k=10)
    assert cands[0]["similarity"] == 1.0              # a: (2/11) / (2/11)
    assert cands[1]["similarity"] == round(11 / 24, 4)  # b: (1/12) / (2/11)


def test_rrf_k_parameter_affects_score():
    """RRF_K 参数生效：不同 k 得到不同的归一化相似度。"""
    vec = [hit("a"), hit("b")]
    bm25 = [hit("a")]
    cands_10, _ = rag._rrf_fuse(vec, bm25, k=10)
    cands_100, _ = rag._rrf_fuse(vec, bm25, k=100)
    assert cands_10[1]["similarity"] == round(11 / 24, 4)
    assert cands_10[1]["similarity"] != cands_100[1]["similarity"]


def test_rrf_similarity_normalized():
    """所有 similarity 落在 [0,1]，双路第 1 归一化为 1.0。"""
    vec = [hit("a"), hit("b"), hit("c")]
    bm25 = [hit("a"), hit("b"), hit("c")]
    cands, _ = rag._rrf_fuse(vec, bm25, k=60)
    assert all(0.0 <= c["similarity"] <= 1.0 for c in cands)
    assert cands[0]["similarity"] == 1.0


def test_rrf_source_label():
    """source 标签覆盖：双路命中 / 仅向量 / 仅BM25 / 无。"""
    assert rag._rrf_fuse([hit("a")], [hit("a")], k=60)[1] == "双路命中"
    assert rag._rrf_fuse([hit("a")], [hit("c")], k=60)[1] == "仅向量"
    assert rag._rrf_fuse([], [hit("a")], k=60)[1] == "仅BM25"
    assert rag._rrf_fuse([], [], k=60)[1] == "无"
