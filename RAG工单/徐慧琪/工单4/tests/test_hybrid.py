# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import pytest

from rag04.config import Settings
from rag04.schema import Chunk, Hit
from rag04.retrieve.hybrid import (
    rrf_fuse, is_image_pointing, apply_image_boost, hybrid_retrieve,
)


def _h(cid, score=1.0, bt="text", page=1, dedup=""):
    return Hit(chunk_id=cid, doc_id="d", page=page, block_type=bt,
               source_id=cid, text=f"text-{cid}", score=score, dedup_key=dedup)


def test_rrf_rewards_items_in_multiple_lists():
    a = [_h("x"), _h("y")]
    b = [_h("y"), _h("z")]
    fused = rrf_fuse([a, b], k=60)
    assert fused[0].chunk_id == "y", "同时出现在两路的块应排第一"


def test_rrf_scores_descending():
    fused = rrf_fuse([[_h("a"), _h("b"), _h("c")]], k=60)
    assert [f.score for f in fused] == sorted([f.score for f in fused], reverse=True)


def test_rrf_weights_shift_ranking():
    a = [_h("a1"), _h("a2")]
    b = [_h("b1"), _h("b2")]
    fused = rrf_fuse([a, b], k=60, weights=[3.0, 1.0])
    assert fused[0].chunk_id.startswith("a")


def test_rrf_empty_inputs():
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []


def test_rrf_dedups_by_chunk_id():
    fused = rrf_fuse([[_h("a"), _h("a")]], k=60)
    assert len([f for f in fused if f.chunk_id == "a"]) == 1


# --- RC2：同一张图的「图像命中」与「描述文本命中」在融合中只占一席 ---

def test_rrf_merges_figure_image_and_description_by_dedup_key():
    """图描述文本块（dedup_key=image 块 id）与 CLIP 图像命中必须合并成一席，
    且代表命中是图像块（保留 image_path 与图像指向加权）。"""
    txt = _h("figtxt", dedup="figimg")
    img = _h("figimg", bt="image")
    fused = rrf_fuse([[txt], [img]], k=60)
    assert len(fused) == 1, f"同图不得重复占位：{[f.chunk_id for f in fused]}"
    assert fused[0].chunk_id == "figimg" and fused[0].block_type == "image"
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 61), "两路贡献应合并累加"


def test_rrf_keeps_different_figures_separate():
    a = _h("figa", bt="image")
    b = _h("figb", bt="image")
    fused = rrf_fuse([[a], [b]], k=60)
    assert {f.chunk_id for f in fused} == {"figa", "figb"}


def test_hybrid_dedups_figure_channels_end_to_end():
    """真实入口验证：dense 文本命中与 CLIP 图像命中同图时融合只出一条。"""
    from rag04.ingest.store import COLL_TEXT
    s = Settings(pipeline_mode="full_04")
    store = _FakeStore(hits={COLL_TEXT: [_h("figtxt", dedup="figimg")]})

    def clip_fn(question, s_, st, k=30):
        return [_h("figimg", bt="image")]

    out = hybrid_retrieve(_PLAIN_Q, s, store, _FakeBM25(), lambda q, s_: [0.0],
                          clip_fn=clip_fn, k_each=5)
    assert [h.chunk_id for h in out] == ["figimg"]


def test_is_image_pointing_detects_chart_words():
    for q in ["组织结构图中销售部有几个部门",
              "从2008年中国IC市场应用结构与增长图中可以看出",
              "请看下图", "如图所示"]:
        assert is_image_pointing(q), f"应判为图像指向：{q}"


def test_is_image_pointing_false_for_plain():
    for q in ["本次发行股数是多少", "法定代表人是谁"]:
        assert not is_image_pointing(q), f"不应判为图像指向：{q}"


def test_apply_image_boost_promotes_image_hits():
    hits = [_h("t1", 1.0, "text"), _h("i1", 0.9, "image")]
    out = apply_image_boost(hits, "组织结构图中销售部有几个部门", boost=1.5)
    assert out[0].chunk_id == "i1", "图像指向问句应提升图像块"


def test_apply_image_boost_noop_for_plain_question():
    hits = [_h("t1", 1.0, "text"), _h("i1", 0.9, "image")]
    out = apply_image_boost(hits, "法定代表人是谁", boost=1.5)
    assert out[0].chunk_id == "t1"


# --- hybrid_retrieve：容错隔离、全失败降级、baseline_03 CLIP 跳过 ---

_PLAIN_Q = "本次发行股数是多少"            # 非图像指向
_IMAGE_Q = "组织结构图中销售部有几个部门"  # 图像指向


def _chunk(cid, bt="text"):
    return Chunk(chunk_id=cid, doc_id="d", page=1, block_type=bt,
                 source_id=cid, text=f"text-{cid}")


class _FakeStore:
    """只需实现 hybrid_retrieve 用到的 search。raises=True 模拟存储故障。"""

    def __init__(self, hits=None, raises=False):
        self.hits = hits or {}
        self.raises = raises
        self.calls = []

    def search(self, coll, vector, k=30):
        self.calls.append({"coll": coll, "k": k})
        if self.raises:
            raise RuntimeError("store-boom")
        return list(self.hits.get(coll, []))[:k]


class _FakeBM25:
    """只需实现 hybrid_retrieve 用到的 search / chunks。"""

    def __init__(self, chunks=(), pairs=(), raises=False):
        self.chunks = list(chunks)
        self.pairs = list(pairs)
        self.raises = raises
        self.calls = []

    def search(self, query, k=30, exclude_ids=None):
        self.calls.append({"query": query, "k": k, "exclude_ids": exclude_ids})
        if self.raises:
            raise RuntimeError("bm25-boom")
        pairs = [(cid, sc) for cid, sc in self.pairs
                 if exclude_ids is None or cid not in exclude_ids]
        return pairs[:k]


@pytest.mark.parametrize("fail_path, survivors", [
    ("dense", {"sparse-1", "clip-1"}),
    ("sparse", {"dense-1", "clip-1"}),
    ("clip", {"dense-1", "sparse-1"}),
])
def test_hybrid_retrieve_isolates_single_path_failure(fail_path, survivors):
    """任一路抛异常，其余各路仍须贡献命中（工单容错硬要求）。"""
    store = _FakeStore(
        hits={"text_chunks": [_h("dense-1")], "table_chunks": []},
        raises=(fail_path == "dense"),
    )
    bm25 = _FakeBM25(
        chunks=[_chunk("sparse-1")], pairs=[("sparse-1", 3.0)],
        raises=(fail_path == "sparse"),
    )

    def clip_fn(question, s, st, k=30):
        if fail_path == "clip":
            raise RuntimeError("clip-boom")
        return [_h("clip-1", bt="image")]

    out = hybrid_retrieve(
        _PLAIN_Q, Settings(pipeline_mode="full_04"), store, bm25,
        lambda q, s: [0.0], clip_fn=clip_fn, k_each=5,
    )
    assert {h.chunk_id for h in out} == survivors


def test_hybrid_retrieve_all_paths_fail_returns_empty():
    """三路全部抛异常：返回 []，且不向上抛。"""
    def clip_fn(question, s, store, k=30):
        raise RuntimeError("clip-boom")

    out = hybrid_retrieve(
        _PLAIN_Q, Settings(pipeline_mode="full_04"),
        _FakeStore(raises=True), _FakeBM25(raises=True),
        lambda q, s: [0.0], clip_fn=clip_fn, k_each=5,
    )
    assert out == []


# --- RC3：图像指向时融合结果必须给文本保留重排席位（rank ≤ rerank_top_n） ---

def test_image_pointing_caps_clip_channel_k():
    """图像指向时 CLIP 路 k 收敛到 clip_max_k，避免 30 个重排名额被图像占满。"""
    calls = []

    def clip_fn(question, s_, store, k=30):
        calls.append(k)
        return [_h(f"i{i}", bt="image") for i in range(k)]

    s = Settings(pipeline_mode="full_04")
    hybrid_retrieve(_IMAGE_Q, s, _FakeStore(), _FakeBM25(),
                    lambda q, s_: [0.0], clip_fn=clip_fn, k_each=30)
    assert calls == [s.clip_max_k]
    assert s.clip_max_k < s.rerank_top_n, "CLIP 配额必须小于重排窗口才有文本席位"


def test_plain_question_keeps_full_clip_channel_k():
    calls = []

    def clip_fn(question, s_, store, k=30):
        calls.append(k)
        return [_h(f"i{i}", bt="image") for i in range(k)]

    hybrid_retrieve(_PLAIN_Q, Settings(pipeline_mode="full_04"),
                    _FakeStore(), _FakeBM25(), lambda q, s_: [0.0],
                    clip_fn=clip_fn, k_each=30)
    assert calls == [30], "非图像指向时 CLIP 路仍取满 k_each（权重 1.0，不构成锁死）"


def test_image_fusion_keeps_text_in_rerank_window():
    """RC3 核心：图像指向时融合 top-30 必须含文本块（金标文本才有机会进重排）。"""
    s = Settings(pipeline_mode="full_04")
    store = _FakeStore(hits={"text_chunks": [_h(f"t{i}") for i in range(30)],
                             "table_chunks": []})

    def clip_fn(question, s_, st, k=30):
        return [_h(f"i{i}", bt="image") for i in range(k)]

    out = hybrid_retrieve(_IMAGE_Q, s, store, _FakeBM25(), lambda q, s_: [0.0],
                          clip_fn=clip_fn, k_each=30)
    top = out[:s.rerank_top_n]
    n_img = sum(1 for h in top if h.block_type == "image")
    n_text = sum(1 for h in top if h.block_type == "text")
    assert n_img <= s.clip_max_k, f"图像最多占 {s.clip_max_k} 席，实测 {n_img}"
    assert n_text == s.rerank_top_n - n_img, "其余席位必须留给文本"


def test_image_fusion_hard_caps_clip_even_if_channel_overreturns():
    """CLIP 实现不守 k 时也硬截断：席位保证不能依赖被调方的自觉。"""
    s = Settings(pipeline_mode="full_04")

    def greedy_clip(question, s_, st, k=30):
        return [_h(f"i{i}", bt="image") for i in range(30)]

    store = _FakeStore(hits={"text_chunks": [_h(f"t{i}") for i in range(30)],
                             "table_chunks": []})
    out = hybrid_retrieve(_IMAGE_Q, s, store, _FakeBM25(), lambda q, s_: [0.0],
                          clip_fn=greedy_clip, k_each=30)
    n_img = sum(1 for h in out[:s.rerank_top_n] if h.block_type == "image")
    assert n_img <= s.clip_max_k


# --- RC1：样板过滤一致作用于三通道（dense / BM25 / CLIP 文本侧） ---

def test_boilerplate_excluded_from_all_three_channels():
    from rag04.ingest.store import COLL_TEXT, COLL_TABLE
    s = Settings(pipeline_mode="full_04")
    bp = {"dense-bp", "table-bp", "sparse-bp", "clip-bp"}
    store = _FakeStore(hits={
        COLL_TEXT: [_h("dense-bp"), _h("dense-keep")],
        COLL_TABLE: [_h("table-bp", bt="table"), _h("table-keep", bt="table")],
    })
    bm25 = _FakeBM25(chunks=[_chunk("sparse-bp"), _chunk("sparse-keep")],
                     pairs=[("sparse-bp", 9.0), ("sparse-keep", 8.0)])

    def clip_fn(question, s_, st, k=30):
        return [_h("clip-bp", bt="image"), _h("clip-keep", bt="image")]

    out = hybrid_retrieve(_PLAIN_Q, s, store, bm25, lambda q, s_: [0.0],
                          clip_fn=clip_fn, k_each=5, boilerplate_ids=bp)
    assert {h.chunk_id for h in out} == {"dense-keep", "table-keep",
                                         "sparse-keep", "clip-keep"}
    assert bm25.calls[0]["exclude_ids"] == bp, "稀疏通道须在截断前感知样板集合"


def test_boilerplate_triggers_denser_fetch():
    """过滤会吃掉候选，须按 boilerplate_fetch_k 加深 dense 召回补席位。"""
    from rag04.ingest.store import COLL_TEXT
    s = Settings(pipeline_mode="full_04")
    store = _FakeStore(hits={COLL_TEXT: [_h(f"t{i}") for i in range(50)]})
    hybrid_retrieve(_PLAIN_Q, s, store, _FakeBM25(), lambda q, s_: [0.0],
                    clip_fn=None, k_each=5, boilerplate_ids={"t0"})
    assert store.calls[0]["k"] == s.boilerplate_fetch_k > 5

    store2 = _FakeStore(hits={COLL_TEXT: [_h(f"t{i}") for i in range(50)]})
    hybrid_retrieve(_PLAIN_Q, s, store2, _FakeBM25(), lambda q, s_: [0.0],
                    clip_fn=None, k_each=5)
    assert store2.calls[0]["k"] == 5, "无样板过滤时不得加固定深挖开销"


def test_boilerplate_filter_keeps_k_each_after_refill():
    """深挖 + 过滤后每路仍取满 k_each，融合规模不缩水。"""
    from rag04.ingest.store import COLL_TEXT
    s = Settings(pipeline_mode="full_04")
    hits = [_h("bp")] + [_h(f"t{i}") for i in range(40)]
    store = _FakeStore(hits={COLL_TEXT: hits})
    out = hybrid_retrieve(_PLAIN_Q, s, store, _FakeBM25(), lambda q, s_: [0.0],
                          clip_fn=None, k_each=5, boilerplate_ids={"bp"})
    assert len(out) == 5 and all(h.chunk_id != "bp" for h in out)


# --- Fix 7：use_hybrid 是双模式总开关（baseline 必须只跑纯稠密） ---

def test_use_hybrid_false_skips_sparse_route():
    """baseline_03（use_hybrid=False）不得跑 BM25：计划的双模式语义是纯稠密。

    原实现无条件跑稀疏路，baseline 实际执行的是 dense+sparse 的 RRF。
    """
    s = Settings(pipeline_mode="baseline_03")
    assert s.use_hybrid is False
    bm25 = _FakeBM25(chunks=[_chunk("sparse-1")], pairs=[("sparse-1", 9.0)])
    store = _FakeStore(hits={"text_chunks": [_h("dense-1")], "table_chunks": []})

    out = hybrid_retrieve(_PLAIN_Q, s, store, bm25, lambda q, s_: [0.0],
                          clip_fn=None, k_each=5)
    assert bm25.calls == [], "use_hybrid=False 时不得调用稀疏检索"
    assert {h.chunk_id for h in out} == {"dense-1"}


def test_use_hybrid_true_keeps_sparse_route():
    """对照：full_04（use_hybrid=True）稀疏路照常参与融合。"""
    s = Settings(pipeline_mode="full_04")
    bm25 = _FakeBM25(chunks=[_chunk("sparse-1")], pairs=[("sparse-1", 9.0)])
    store = _FakeStore(hits={"text_chunks": [_h("dense-1")], "table_chunks": []})
    out = hybrid_retrieve(_PLAIN_Q, s, store, bm25, lambda q, s_: [0.0],
                          clip_fn=None, k_each=5)
    assert len(bm25.calls) == 1
    assert {h.chunk_id for h in out} == {"dense-1", "sparse-1"}


def test_use_hybrid_switch_is_independent_of_clip():
    """use_hybrid 只管稀疏路：CLIP 仍由 use_clip_retrieval 决定（各开关正交）。"""
    s = Settings(pipeline_mode="full_04", use_hybrid=False)
    calls = []

    def clip_fn(question, s_, store, k=30):
        calls.append(k)
        return [_h("clip-1", bt="image")]

    store = _FakeStore(hits={"text_chunks": [_h("dense-1")], "table_chunks": []})
    out = hybrid_retrieve(_PLAIN_Q, s, store, _FakeBM25(), lambda q, s_: [0.0],
                          clip_fn=clip_fn, k_each=5)
    assert calls and {h.chunk_id for h in out} == {"dense-1", "clip-1"}


# --- Fix 8：空召回/空索引必须留下 WARNING（原先只有 CLIP 路有日志） ---

def test_empty_dense_and_sparse_routes_log_warning(caplog):
    """三路皆空时：稠密空召回、稀疏空索引、融合空结果都要有可排查的告警。"""
    import logging

    s = Settings(pipeline_mode="full_04")
    store = _FakeStore(hits={"text_chunks": [], "table_chunks": []})
    with caplog.at_level(logging.WARNING, logger="rag04.hybrid"):
        out = hybrid_retrieve(_PLAIN_Q, s, store, _FakeBM25(), lambda q, s_: [0.0],
                              clip_fn=None, k_each=5)

    assert out == []
    text = caplog.text
    assert "稠密召回 0 命中" in text
    assert "稀疏索引为空" in text, "空 BM25 索引须点名原因"
    assert "融合结果为 0 命中" in text


def test_sparse_route_with_no_token_overlap_names_other_cause(caplog):
    """BM25 索引非空但查询词无命中：原因文案不得写成「索引为空」。"""
    import logging

    s = Settings(pipeline_mode="full_04")
    bm25 = _FakeBM25(chunks=[_chunk("c1")], pairs=[])
    store = _FakeStore(hits={"text_chunks": [_h("dense-1")], "table_chunks": []})
    with caplog.at_level(logging.WARNING, logger="rag04.hybrid"):
        hybrid_retrieve(_PLAIN_Q, s, store, bm25, lambda q, s_: [0.0],
                        clip_fn=None, k_each=5)

    assert "查询词元未命中任何块" in caplog.text
    assert "稀疏索引为空" not in caplog.text


def test_empty_clip_route_logs_warning(caplog):
    """CLIP 路 0 命中原先只记 INFO，统一为 WARNING 以便压测排查。"""
    import logging

    s = Settings(pipeline_mode="full_04")
    store = _FakeStore(hits={"text_chunks": [_h("dense-1")], "table_chunks": []})
    with caplog.at_level(logging.WARNING, logger="rag04.hybrid"):
        hybrid_retrieve(_PLAIN_Q, s, store, _FakeBM25(),
                        lambda q, s_: [0.0],
                        clip_fn=lambda q, s_, st, k=30: [], k_each=5)
    assert "CLIP 跨模态召回 0 命中" in caplog.text


def test_healthy_retrieval_logs_no_empty_warnings(caplog):
    """有命中时不得刷空召回告警（避免评估/压测日志噪声）。"""
    import logging

    s = Settings(pipeline_mode="full_04")
    bm25 = _FakeBM25(chunks=[_chunk("sparse-1")], pairs=[("sparse-1", 3.0)])
    store = _FakeStore(hits={"text_chunks": [_h("dense-1")], "table_chunks": []})
    with caplog.at_level(logging.WARNING, logger="rag04.hybrid"):
        hybrid_retrieve(_PLAIN_Q, s, store, bm25, lambda q, s_: [0.0],
                        clip_fn=None, k_each=5)
    assert caplog.text == "", f"正常召回不得有告警：{caplog.text}"


def test_hybrid_retrieve_baseline_skips_clip_channel():
    """baseline_03 控制条件：use_clip_retrieval=False 时 clip_fn 绝不被调用。"""
    s = Settings(pipeline_mode="baseline_03")
    assert s.use_clip_retrieval is False, "基座模式下 CLIP 开关应为关"

    calls = {"n": 0}

    def clip_fn(question, s_, store, k=30):
        calls["n"] += 1
        raise AssertionError("baseline_03 不应调用 CLIP 通道")

    store = _FakeStore(hits={"text_chunks": [_h("dense-1")],
                             "table_chunks": []})
    out = hybrid_retrieve(
        _IMAGE_Q, s, store, _FakeBM25(), lambda q, s_: [0.0],
        clip_fn=clip_fn, k_each=5,
    )
    assert calls["n"] == 0
    assert {h.chunk_id for h in out} == {"dense-1"}
