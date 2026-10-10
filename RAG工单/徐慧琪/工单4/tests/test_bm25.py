# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
from rag04.schema import Chunk
from rag04.retrieve.bm25 import tokenize_zh, BM25Index


def _c(i, text, page=1, bt="text"):
    return Chunk(chunk_id=f"c{i}", doc_id="d", page=page, block_type=bt,
                 source_id=f"s{i}", text=text)


def test_tokenize_zh_splits_chinese():
    toks = tokenize_zh("武汉力源信息技术股份有限公司")
    assert len(toks) > 1
    assert any("力源" in t for t in toks)


def test_tokenize_zh_keeps_ascii_words():
    toks = tokenize_zh("公司 IC 市场规模 2008")
    assert "ic" in [t.lower() for t in toks] or "IC" in toks


def test_tokenize_zh_empty():
    assert tokenize_zh("") == []


def test_search_ranks_relevant_first():
    idx = BM25Index()
    idx.build([
        _c(0, "销售部下设渠道销售部、电话及网络销售部、大客户销售部和国际贸易部"),
        _c(1, "2008年中国IC市场应用结构与增长图显示汽车电子增长率最快"),
        _c(2, "本次发行股数占发行后总股本的比例"),
    ])
    hits = idx.search("大客户销售部", k=3)
    assert hits
    assert hits[0][0] == "c0", f"最相关块应排第一，实际={hits}"


def test_search_returns_empty_for_no_match():
    idx = BM25Index()
    idx.build([_c(0, "武汉力源信息技术")])
    assert idx.search("完全无关的查询词xyz", k=5) == []


def test_search_respects_k():
    idx = BM25Index()
    idx.build([_c(i, f"武汉力源信息 第{i}段") for i in range(20)])
    assert len(idx.search("武汉力源", k=5)) <= 5


def test_save_and_load_roundtrip(tmp_path):
    idx = BM25Index()
    idx.build([_c(0, "销售部组织结构图"), _c(1, "募集资金投资项目")])
    p = tmp_path / "bm25.pkl"
    idx.save(p)
    idx2 = BM25Index.load(p)
    assert len(idx2.chunks) == 2
    assert idx2.search("销售部", k=1)[0][0] == "c0"


def test_search_exclude_ids_filters_before_top_k():
    """RC1：排除样板必须在截取 top-k 之前生效，否则 top-30 全是页眉时会被截空。"""
    idx = BM25Index()
    idx.build([_c(i, "招股意向书 页眉样板文本") for i in range(5)]
              + [_c(99, "页眉样板文本 之后的正文候选")])

    all_hits = idx.search("页眉样板文本", k=10)
    assert len(all_hits) == 6
    excluded = {f"c{i}" for i in range(5)}
    rest = idx.search("页眉样板文本", k=10, exclude_ids=excluded)
    assert [cid for cid, _ in rest] == ["c99"], "排除后仍应命中原池更深的块"

    top1 = idx.search("页眉样板文本", k=1, exclude_ids={"c4"})
    assert top1 and top1[0][0] != "c4", "先排除再截断：k=1 也应跳过被排除的块"


def test_search_exclude_ids_none_is_noop():
    idx = BM25Index()
    idx.build([_c(0, "销售部组织结构图")])
    assert idx.search("销售部", k=5, exclude_ids=None) == idx.search("销售部", k=5)


def test_scores_are_descending():
    idx = BM25Index()
    idx.build([_c(0, "销售部 销售部 销售部"), _c(1, "销售部"), _c(2, "无关")])
    hits = idx.search("销售部", k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
