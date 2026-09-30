from app.core.retrieve.hybrid_retriever import BM25Index, rrf_fusion, tokenize


def test_tokenize_chinese():
    # 中文必须整词切（jieba）：按字切会让 IDF 失真，专业术语几乎召不回来。
    tokens = tokenize("高血压患者饮食")
    assert "高血压" in tokens


def test_rrf_fusion_ranks_common_item_first():
    ranked = rrf_fusion(["x", "a"], ["y", "x"])
    ids = [i for i, _ in ranked]
    assert ids[0] == "x"  # x 同时出现在两路召回，融合分最高


def test_bm25_returns_matching_docs_only():
    # 无关块一条都不许进：它们会占掉 TOP_K，还把无关内容顶进 prompt。
    items = [
        {"id": "1", "text": "高血压患者应限制钠盐摄入"},
        {"id": "2", "text": "苹果富含维生素"},
        {"id": "3", "text": "高血压患者应规律运动"},
    ]
    idx = BM25Index(items)
    hits = idx.search("高血压", top_k=5)
    hit_ids = [h["id"] for h in hits]
    assert "1" in hit_ids and "3" in hit_ids
    assert "2" not in hit_ids


def test_bm25_empty_corpus():
    # 空库要返回空而不是抛：第一份文档上传之前就会走到检索。
    idx = BM25Index([])
    assert idx.search("高血压", top_k=5) == []


def test_bm25_keeps_hits_when_idf_is_exactly_zero():
    """df == N/2 时 rank_bm25 的平滑 IDF 恰为 ln(1)=0：**命中文档的分数就是 0.0**。

    所以「score == 0 表示词没出现」这个前提不成立。旧实现先按分数降序切片、再拿
    `!= 0.0` 过滤，这种语料会被整路滤空——实测（4 块里 2 块含「苹果」）返回 0 条，
    而稠密路察觉不到，两路召回静默不对齐。判据必须换成「query 的 token 是否真的出现在该块里」。
    """
    texts = ["苹果的营养价值很高", "苹果适合做沙拉", "今天天气不错", "明天也要出门"]
    idx = BM25Index([{"id": str(i), "text": t} for i, t in enumerate(texts)])

    assert idx.bm25.get_scores(tokenize("苹果")).max() == 0.0, "前提：这种语料下所有分数都是 0"
    assert {h["id"] for h in idx.search("苹果", top_k=5)} == {"0", "1"}


def test_hybrid_retriever_bm25_cache_invalidated_after_new_data():
    """入库后 BM25 缓存必须失效，否则新 chunk 永远进不了关键词召回。

    覆盖「上传 → invalidate → 新数据可召回」这条链路：未失效前走的是旧索引，
    失效后才重建。之前 invalidate 加锁时顺手补的回归，防止缓存语义被改坏。
    """
    from app.core.config import Settings
    from app.core.retrieve.hybrid_retriever import HybridRetriever

    class _MutableMilvus:
        def __init__(self):
            self.texts: list[dict] = []

        def search(self, role_id, qv, top_k):
            return []  # 稠密路无命中，聚焦验证 BM25 路径

        def list_texts(self, role_id, limit=10000):
            return self.texts

    class _DummyEmbed:
        def embed_query(self, text):
            return [0.0] * 10

    mv = _MutableMilvus()
    ret = HybridRetriever(Settings(top_k=3, score_threshold=0.0), _DummyEmbed(), mv)

    # 空库：首次检索构建了空索引
    assert ret.retrieve("高血压", "doctor") == []

    # 模拟入库：Milvus 里新增一条含「高血压」的 chunk
    mv.texts.append(
        {"id": "n1", "text": "高血压患者应限制钠盐摄入", "title": "t", "source": "s", "chunk_index": 0}
    )

    # 未失效：BM25 仍是旧索引，召不回新数据
    assert ret.retrieve("高血压", "doctor") == []

    ret.invalidate("doctor")
    hits = ret.retrieve("高血压", "doctor")
    assert [h["id"] for h in hits] == ["n1"], f"失效后应召回新 chunk，实际 {hits}"
