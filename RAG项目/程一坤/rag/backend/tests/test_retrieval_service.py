"""测试检索服务总入口。

任务书 3-C 要求：
- 串起完整检索链路：过滤 → 向量召回 + 关键词召回 → RRF 融合 → 重排 → 上下文组装
- 返回：候选条文 + 引用元数据 + 分数明细
- 能看到两路召回条数、融合后条数、重排后条数
"""

import logging

from app.models.reranker import RankedCandidate
from app.retrieval.service import RetrievalService, RetrievalResult
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.fusion import RankedItem


def test_retrieval_service_initialization():
    """测试服务初始化"""
    # 使用 mock 依赖
    service = RetrievalService(
        vector_retriever=None,  # 暂时用 None，后续用 mock
        keyword_searcher=None,
        session_factory=None,
    )
    assert service is not None


def test_retrieval_result_structure():
    """测试返回结果结构：包含候选条文、分数明细"""
    result = RetrievalResult(
        articles=[
            RetrievedArticle(
                chunk_key="chunk1",
                content="测试内容",
                article_number="1",
                document_title="测试法规",
                source_url="https://example.com",
                recall_score=0.9,
                rerank_score=0.85,
                document_type="法律",
                jurisdiction="中国大陆",
                is_current=True,
            )
        ],
        context_block="[1] 《测试法规》第 1 条\n测试内容",
        stats={
            "vector_recall_count": 10,
            "keyword_recall_count": 8,
            "fused_count": 15,
            "reranked_count": 5,
        },
    )

    assert len(result.articles) == 1
    assert result.stats["vector_recall_count"] == 10
    assert result.stats["keyword_recall_count"] == 8
    assert result.stats["fused_count"] == 15
    assert result.stats["reranked_count"] == 5
    assert "[1]" in result.context_block


def test_retrieval_stats_visibility():
    """测试分数明细可见：两路召回条数、融合后条数、重排后条数"""
    stats = {
        "vector_recall_count": 20,
        "keyword_recall_count": 15,
        "fused_count": 28,  # 去重后
        "reranked_count": 5,
    }

    # 验证字段存在
    assert "vector_recall_count" in stats
    assert "keyword_recall_count" in stats
    assert "fused_count" in stats
    assert "reranked_count" in stats

    # 验证逻辑合理：融合后 <= 两路之和，重排后 <= 融合后
    assert stats["fused_count"] <= stats["vector_recall_count"] + stats["keyword_recall_count"]
    assert stats["reranked_count"] <= stats["fused_count"]


class FakeVectorRetriever:
    """返回固定向量候选，并记录服务层是否禁用了向量路重排。"""

    def __init__(self) -> None:
        self.rerank_top_n_values: list[int] = []

    def retrieve(self, question: str, *, rerank_top_n: int, **kwargs):
        self.rerank_top_n_values.append(rerank_top_n)
        return [
            RetrievedArticle("chunk-a", "候选 A", "法规 A", "url-a", 0.9, "第一条"),
            RetrievedArticle("chunk-b", "候选 B", "法规 B", "url-b", 0.8, "第二条"),
        ]


class ReversingReranker:
    """把第二条排到第一条，证明融合后确实调用重排。"""

    def rerank(self, query: str, documents: list[str], top_n: int):
        return [RankedCandidate(index=1, score=0.94), RankedCandidate(index=0, score=0.73)][:top_n]


class FailingReranker:
    def rerank(self, query: str, documents: list[str], top_n: int):
        raise RuntimeError("模拟重排不可用")


class RecordingKeywordSearcher:
    """记录关键词路是否按开关执行。"""

    def __init__(self) -> None:
        self.search_count = 0

    def search(self, question: str, top_k: int, **kwargs):
        self.search_count += 1
        return []


def test_reranks_fused_candidates_and_separates_scores() -> None:
    vector_retriever = FakeVectorRetriever()
    service = RetrievalService(
        vector_retriever=vector_retriever,
        reranker=ReversingReranker(),
    )

    result = service.retrieve("经济补偿", vector_recall_limit=20, rerank_top_n=2)

    assert [article.chunk_key for article in result.articles] == ["chunk-b", "chunk-a"]
    assert result.articles[0].rerank_score == 0.94
    assert result.articles[0].fusion_score is not None
    assert result.articles[0].rerank_score != result.articles[0].fusion_score
    assert vector_retriever.rerank_top_n_values == [0]


def test_reranker_failure_logs_warning_and_falls_back(caplog) -> None:
    service = RetrievalService(
        vector_retriever=FakeVectorRetriever(),
        reranker=FailingReranker(),
    )

    with caplog.at_level(logging.WARNING):
        result = service.retrieve("经济补偿", rerank_top_n=2, request_id="req_test123456")

    assert [article.chunk_key for article in result.articles] == ["chunk-a", "chunk-b"]
    assert all(article.rerank_score is None for article in result.articles)
    assert result.stats["rerank_fallback"] is True
    assert result.stats["rerank_error_type"] == "RuntimeError"
    warning = next(record for record in caplog.records if "重排服务不可用" in record.message)
    assert warning.request_id == "req_test123456"


def test_hybrid_and_rerank_are_enabled_by_default() -> None:
    keyword_searcher = RecordingKeywordSearcher()
    service = RetrievalService(
        vector_retriever=FakeVectorRetriever(),
        keyword_searcher=keyword_searcher,
        session_factory=object(),
        reranker=ReversingReranker(),
    )

    result = service.retrieve("经济补偿", rerank_top_n=2)

    assert keyword_searcher.search_count == 1
    assert [article.chunk_key for article in result.articles] == ["chunk-b", "chunk-a"]


def test_hybrid_and_rerank_can_be_disabled() -> None:
    keyword_searcher = RecordingKeywordSearcher()
    service = RetrievalService(
        vector_retriever=FakeVectorRetriever(),
        keyword_searcher=keyword_searcher,
        session_factory=object(),
        reranker=ReversingReranker(),
    )

    result = service.retrieve(
        "经济补偿",
        rerank_top_n=2,
        enable_hybrid_search=False,
        enable_rerank=False,
    )

    assert keyword_searcher.search_count == 0
    assert [article.chunk_key for article in result.articles] == ["chunk-a", "chunk-b"]
    assert all(article.rerank_score is None for article in result.articles)


def test_preserves_citation_metadata_through_fusion() -> None:
    class MetadataVectorRetriever:
        def retrieve(self, question: str, **kwargs):
            return [
                RetrievedArticle(
                    chunk_key="law-a47",
                    content="第四十七条正文",
                    document_title="中华人民共和国劳动合同法",
                    source_url="https://example.gov.cn/law",
                    recall_score=0.9,
                    article_number="第四十七条",
                    paragraph_number="1",
                    item_number="2",
                    document_type="法律",
                    jurisdiction="中国大陆",
                    effective_date=1356969600,
                    expiration_date=0,
                    issuing_authority="全国人民代表大会常务委员会",
                    is_current=True,
                )
            ]

    service = RetrievalService(vector_retriever=MetadataVectorRetriever())

    result = service.retrieve("经济补偿", rerank_top_n=1, enable_rerank=False)

    article = result.articles[0]
    assert article.document_type == "法律"
    assert article.jurisdiction == "中国大陆"
    assert article.effective_date == 1356969600
    assert article.expiration_date == 0
    assert article.issuing_authority == "全国人民代表大会常务委员会"
    assert article.is_current is True
    assert article.paragraph_number == "1"
    assert article.item_number == "2"


def test_collapses_child_chunks_to_parent_and_keeps_score_sources() -> None:
    service = RetrievalService()
    reranked = [
        RankedItem(
            chunk_key="law-a47-c1",
            parent_chunk_key="law-a47",
            score=0.94,
            source="rerank",
            sources=("vector",),
            content="第四十七条完整正文",
            article_number="第四十七条",
            rerank_score=0.94,
            vector_score=0.74,
        ),
        RankedItem(
            chunk_key="law-a47-c2",
            parent_chunk_key="law-a47",
            score=0.73,
            source="rerank",
            sources=("keyword",),
            content="第四十七条完整正文",
            article_number="第四十七条",
            rerank_score=0.73,
            keyword_score=7.35,
        ),
        RankedItem(
            chunk_key="law-a46",
            parent_chunk_key=None,
            score=0.60,
            source="rerank",
            sources=("vector", "keyword"),
            content="第四十六条正文",
            article_number="第四十六条",
            rerank_score=0.60,
            vector_score=0.68,
            keyword_score=5.12,
        ),
    ]

    collapsed = service._collapse_parent_chunks(reranked, top_n=8)

    assert [item.chunk_key for item in collapsed] == ["law-a47", "law-a46"]
    assert collapsed[0].rerank_score == 0.94
    assert collapsed[0].vector_score == 0.74
    assert collapsed[0].keyword_score == 7.35
    assert set(collapsed[0].sources) == {"vector", "keyword"}


def test_deduplicates_same_parent_before_rerank() -> None:
    """批次 12-A：父块归并提前到重排之前，同父块子块不再重复占重排名额。"""
    from types import SimpleNamespace

    def art(key: str, parent: str, content: str, score: float) -> SimpleNamespace:
        return SimpleNamespace(
            chunk_key=key,
            parent_chunk_key=parent,
            content=content,
            document_title="中华人民共和国劳动合同法",
            article_number="第四十七条" if "a47" in key else "第四十六条",
            source_url="u",
            recall_score=score,
            rerank_score=None,
            paragraph_number=None,
            item_number=None,
            document_type="法律",
            jurisdiction="中国大陆",
            effective_date=0,
            expiration_date=0,
            issuing_authority="x",
            is_current=True,
            document_id="d1",
        )

    class DuplicateVectorRetriever:
        """向量路返回同父块两个子块 + 另一父块一条。"""

        def retrieve(self, question: str, *, rerank_top_n: int, **kwargs):
            return [
                art("law-a47-c1", "law-a47", "第四十七条完整正文", 0.9),
                art("law-a47-c2", "law-a47", "第四十七条完整正文", 0.85),
                art("law-a46", "law-a46", "第四十六条完整正文", 0.8),
            ]

    class RecordingReranker:
        def __init__(self) -> None:
            self.documents: list[str] = []

        def rerank(self, query: str, documents: list[str], top_n: int):
            self.documents = list(documents)
            return [RankedCandidate(index=i, score=0.9 - i * 0.1) for i in range(len(documents))][:top_n]

    reranker = RecordingReranker()
    service = RetrievalService(vector_retriever=DuplicateVectorRetriever(), reranker=reranker)

    result = service.retrieve("经济补偿", rerank_top_n=2)

    # 同一父块只送重排一次：47条归并成1条 + 46条 = 2 条，无重复正文
    assert len(reranker.documents) == 2
    assert len(set(reranker.documents)) == 2
    # 最终结果无重复父块
    keys = [article.chunk_key for article in result.articles]
    assert len(keys) == len(set(keys))


def test_rerank_input_has_law_anchor_prefix() -> None:
    """批次 12-B：重排输入从纯正文改为「《法规名》条号 正文」，给重排模型锚点。"""
    from types import SimpleNamespace

    class AnchorVectorRetriever:
        def retrieve(self, question: str, *, rerank_top_n: int, **kwargs):
            return [SimpleNamespace(
                chunk_key="law-a46",
                parent_chunk_key="law-a46",
                content="用人单位应当在解除或者终止劳动合同时出具证明。",
                document_title="中华人民共和国劳动合同法",
                article_number="第五十条",
                source_url="u",
                recall_score=0.9,
                rerank_score=None,
                paragraph_number=None,
                item_number=None,
                document_type="法律",
                jurisdiction="中国大陆",
                effective_date=0,
                expiration_date=0,
                issuing_authority="x",
                is_current=True,
                document_id="d1",
            )]

    class RecordingReranker:
        def __init__(self) -> None:
            self.documents: list[str] = []

        def rerank(self, query: str, documents: list[str], top_n: int):
            self.documents = list(documents)
            return [RankedCandidate(index=0, score=0.9)][:top_n]

    reranker = RecordingReranker()
    service = RetrievalService(vector_retriever=AnchorVectorRetriever(), reranker=reranker)

    service.retrieve("离职证明", rerank_top_n=1)

    assert reranker.documents == [
        "《中华人民共和国劳动合同法》第五十条 用人单位应当在解除或者终止劳动合同时出具证明。"
    ]


def test_rerank_input_falls_back_without_article_number() -> None:
    """案例类候选没有条号时，锚点只带法规名，正文原样保留。"""
    from types import SimpleNamespace

    class CaseVectorRetriever:
        def retrieve(self, question: str, *, rerank_top_n: int, **kwargs):
            return [SimpleNamespace(
                chunk_key="case-1",
                parent_chunk_key="case-1",
                content="典型案例正文。",
                document_title="最高法发布劳动争议典型案例",
                article_number=None,
                source_url="u",
                recall_score=0.9,
                rerank_score=None,
                paragraph_number=None,
                item_number=None,
                document_type="案例",
                jurisdiction="中国大陆",
                effective_date=0,
                expiration_date=0,
                issuing_authority="x",
                is_current=True,
                document_id="d2",
            )]

    class RecordingReranker:
        def __init__(self) -> None:
            self.documents: list[str] = []

        def rerank(self, query: str, documents: list[str], top_n: int):
            self.documents = list(documents)
            return [RankedCandidate(index=0, score=0.9)][:top_n]

    reranker = RecordingReranker()
    service = RetrievalService(vector_retriever=CaseVectorRetriever(), reranker=reranker)

    service.retrieve("典型案例", rerank_top_n=1)

    assert reranker.documents == ["《最高法发布劳动争议典型案例》 典型案例正文。"]


class IdentityReranker:
    """保持候选顺序并记录收到的并集，验证重试合并发生在重排之前。"""

    def __init__(self) -> None:
        self.documents: list[str] = []

    def rerank(self, query: str, documents: list[str], top_n: int):
        self.documents = documents
        return [
            RankedCandidate(index=index, score=1.0 - index * 0.1)
            for index in range(min(top_n, len(documents)))
        ]


def test_retry_merge_keeps_original_golden_before_rerank() -> None:
    reranker = IdentityReranker()
    service = RetrievalService(reranker=reranker)
    original = RetrievalResult(
        articles=[
            RetrievedArticle(
                "golden-chunk", "原候选 golden 法条", "劳动合同法", "url-golden", 0.2
            ),
            RetrievedArticle(
                "shared-chunk", "原候选重复法条", "劳动合同法", "url-shared", 0.1
            ),
        ],
        context_block="",
        stats={"reranked_count": 2},
    )
    retry = RetrievalResult(
        articles=[
            RetrievedArticle(
                "retry-chunk", "重试候选法条", "劳动合同法", "url-retry", 0.9
            ),
            RetrievedArticle(
                "shared-chunk", "重试重复法条", "劳动合同法", "url-shared", 0.8
            ),
        ],
        context_block="",
        stats={"reranked_count": 2},
    )

    merged = service.merge_retry_results(original, retry, "拼接后的追问", top_n=5)

    assert [article.chunk_key for article in merged.articles] == [
        "golden-chunk", "shared-chunk", "retry-chunk"
    ]
    assert len(reranker.documents) == 3
    assert reranker.documents[1].endswith("原候选重复法条")
    assert merged.stats["retry_merge_original_count"] == 2
    assert merged.stats["retry_merge_count"] == 3
