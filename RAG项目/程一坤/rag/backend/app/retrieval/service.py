"""检索服务总入口：串联完整检索链路。

链路：过滤 → 向量召回 + 关键词召回 → RRF 融合 → 重排 → 上下文组装

对外暴露统一接口，供问答层和 API 层调用。
RankedItem 的字段搬运纯函数在 item_convert.py（本文件只做编排）。
"""
import logging
from dataclasses import dataclass
from typing import Any

from app.retrieval.context_builder import RetrievedArticle, build_context_block
from app.retrieval.filters import build_filter_expression
from app.retrieval.fusion import fuse_results, RankedItem
from app.retrieval.item_convert import (
    _build_rerank_text,
    articles_to_ranked_items,
    ranked_items_to_articles,
    rebuild_rerank_items,
    vector_articles_to_ranked_items,
)
from app.retrieval.keyword_search import KeywordSearcher
from app.retrieval.parent_collapse import collapse_parent_items, load_keyword_parent_items
from app.retrieval.query_rewrite import QueryRewriteResult, QueryRewriter
from app.retrieval.assembly import rewrite_query_safely
from app.retrieval.vector_search import LegalRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果，包含候选条文、上下文和统计信息。

    Attributes:
        articles: 重排后的候选条文列表
        context_block: 格式化的法源清单文本（供大模型使用）
        stats: 统计信息（两路召回条数、融合后条数、重排后条数）
    """
    articles: list[RetrievedArticle]
    context_block: str
    stats: dict[str, Any]
    query_rewrite: QueryRewriteResult | None = None


class RetrievalService:
    """检索服务：整合向量检索和关键词检索。"""

    def __init__(
        self,
        vector_retriever: LegalRetriever | None = None,
        keyword_searcher: KeywordSearcher | None = None,
        session_factory: Any | None = None,
        query_rewriter: QueryRewriter | None = None,
        short_term_memory: Any | None = None,
        reranker: Any | None = None,
    ):
        """初始化检索服务及其可选依赖。"""
        self.vector_retriever = vector_retriever
        self.keyword_searcher = keyword_searcher
        self.session_factory = session_factory
        self.query_rewriter = query_rewriter
        self.short_term_memory = short_term_memory
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        *,
        vector_recall_limit: int | None = None,
        keyword_recall_limit: int | None = None,
        rerank_candidate_limit: int | None = None,
        rerank_top_n: int = 5,
        as_of_date: str | None = None,
        jurisdiction: str = "中国大陆",
        document_types: list[str] | None = None,
        only_current: bool = False,
        max_context_items: int | None = None,
        max_context_length: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        enable_hybrid_search: bool = True,
        enable_rerank: bool = True,
        request_id: str | None = None,
    ) -> RetrievalResult:
        """执行完整检索链路并返回候选、上下文和统计信息。

        召回窗口参数（批次 10 起可配置）：未显式传参时读全局配置
        （RECALL_VECTOR_LIMIT / RECALL_KEYWORD_LIMIT / RERANK_CANDIDATE_LIMIT，
        默认 40/40/40），便于调参而不用改代码。
        """
        from app.core.config import settings

        if vector_recall_limit is None:
            vector_recall_limit = settings.recall_vector_limit
        if keyword_recall_limit is None:
            keyword_recall_limit = settings.recall_keyword_limit
        if rerank_candidate_limit is None:
            rerank_candidate_limit = settings.rerank_candidate_limit
        rewrite_result = rewrite_query_safely(
            question,
            query_rewriter=self.query_rewriter,
            short_term_memory=self.short_term_memory,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )
        retrieval_question = rewrite_result.rewritten_query

        # 第一步：构建过滤表达式
        filter_expr = build_filter_expression(
            as_of_date=as_of_date,
            jurisdiction=jurisdiction,
            document_types=document_types,
            only_current=only_current,
        )

        # 第二步：向量召回
        vector_results = []
        if self.vector_retriever:
            vector_articles = self.vector_retriever.retrieve(
                retrieval_question,
                recall_limit=vector_recall_limit,
                rerank_top_n=0,
                filter_expr=filter_expr,
            )
            # 转换为 RankedItem（字段搬运细节见 item_convert.py）
            vector_results = vector_articles_to_ranked_items(vector_articles)

        # 第三步：关键词召回
        keyword_results = []
        # 同义术语扩写（批次 13，开关默认关）：关键词路会把命中的法条用语
        # 追加进 BM25 查询，这里同步记录条数，供评测对比与回滚核对
        expansion_phrases = self._expansion_phrases(retrieval_question)
        if enable_hybrid_search and self.keyword_searcher and self.session_factory:
            # 先执行关键词检索，获取 KeywordHit 列表
            keyword_hits = self.keyword_searcher.search(
                retrieval_question,
                top_k=keyword_recall_limit,
                jurisdiction=jurisdiction,
                document_types=document_types,
                as_of_date=as_of_date,
                only_current=only_current,
            )

            # 从数据库获取完整的文章信息
            if keyword_hits:
                keyword_results = load_keyword_parent_items(
                    self.session_factory,
                    keyword_hits,
                )

        # 第四步：RRF 融合
        fused_results = fuse_results([vector_results, keyword_results])

        # 第五步（批次 12-A）：先按父块归并，再截断送重排。
        # 旧链路"截断→重排→归并"会让同一父块的多个子块重复占重排名额
        # （实测每题 9~15 个多余名额，把其它条文挤出截断口）。
        # 归并规则不变：同一父块保留最高召回分，子块命中用父块正文。
        premerged = self._collapse_parent_chunks(fused_results, len(fused_results))
        rerank_candidates = premerged[:rerank_candidate_limit]
        rerank_fallback = False
        rerank_error_type = None
        if enable_rerank:
            ranked_items, rerank_fallback, rerank_error_type = self._rerank_fused_candidates(
                retrieval_question,
                rerank_candidates,
                len(rerank_candidates),
                request_id=request_id,
            )
        else:
            ranked_items = rerank_candidates
        # 候选已是父块级唯一，重排后直接按重排分截取最终条数
        collapsed_items = ranked_items[:rerank_top_n]

        # 第六步：转换回 RetrievedArticle
        final_articles = [
            RetrievedArticle(
                chunk_key=item.chunk_key,
                content=item.content or "",
                article_number=item.article_number,
                document_title=item.document_title or "",
                source_url=item.source_url or "",
                recall_score=item.recall_score or item.vector_score or item.keyword_score or 0.0,
                score_sources=item.sources,
                vector_score=item.vector_score,
                keyword_score=item.keyword_score,
                fusion_score=item.fusion_score,
                rerank_score=item.rerank_score,
                paragraph_number=item.paragraph_number,
                item_number=item.item_number,
                document_type=item.document_type,
                jurisdiction=item.jurisdiction,
                effective_date=item.effective_date,
                expiration_date=item.expiration_date,
                issuing_authority=item.issuing_authority,
                is_current=item.is_current,
                document_id=item.document_id,
            )
            for item in collapsed_items
        ]

        # 第七步：组装上下文
        context_block = build_context_block(
            final_articles,
            max_items=max_context_items,
            max_total_length=max_context_length,
        )

        # 统计信息；rerank_fallback 让评测区分真实精排与 RRF 降级，避免把降级轮当作干净基线。
        stats = {
            "vector_recall_count": len(vector_results),
            "keyword_recall_count": len(keyword_results),
            "fused_count": len(fused_results),
            "reranked_count": len(final_articles),
            "synonym_expansion_count": len(expansion_phrases),
            "rerank_fallback": rerank_fallback,
            "rerank_error_type": rerank_error_type,
        }

        return RetrievalResult(
            articles=final_articles,
            context_block=context_block,
            stats=stats,
            query_rewrite=rewrite_result,
        )

    def merge_retry_results(
        self,
        original_result: RetrievalResult,
        retry_result: RetrievalResult,
        question: str,
        *,
        top_n: int,
    ) -> RetrievalResult:
        """合并原始与回退候选，并对并集统一重排。"""
        unique_articles = {}
        for article in [*original_result.articles, *retry_result.articles]:
            unique_articles.setdefault(article.chunk_key, article)
        merged_articles = list(unique_articles.values())
        if not merged_articles:
            return retry_result
        if self.reranker is not None:
            ranked_items, rerank_fallback, rerank_error_type = self._rerank_fused_candidates(
                question,
                articles_to_ranked_items(merged_articles),
                len(merged_articles),
            )
        else:
            ranked_items = articles_to_ranked_items(merged_articles)
            rerank_fallback = True
            rerank_error_type = "not_configured"
        ranked_articles = ranked_items_to_articles(ranked_items[:top_n])
        return RetrievalResult(
            articles=ranked_articles,
            context_block=build_context_block(ranked_articles),
            stats={
                **retry_result.stats,
                "retry_merge_original_count": len(original_result.articles),
                "retry_merge_count": len(merged_articles),
                "rerank_fallback": rerank_fallback,
                "rerank_error_type": rerank_error_type,
            },
            query_rewrite=retry_result.query_rewrite,
        )

    def _rerank_fused_candidates(
        self,
        question: str,
        candidates: list[RankedItem],
        top_n: int,
        request_id: str | None = None,
    ) -> tuple[list[RankedItem], bool, str | None]:
        """精排融合候选；服务不可用时记录告警并退回融合顺序。"""
        if not candidates:
            return [], False, None
        if self.reranker is None:
            logger.warning(
                "未配置重排服务，已退回 RRF 融合顺序",
                extra={"request_id": request_id},
            )
            return candidates[:top_n], True, "not_configured"
        try:
            ranked = self.reranker.rerank(
                question,
                [_build_rerank_text(candidate) for candidate in candidates],
                top_n=top_n,
            )
        except Exception as error:
            logger.warning(
                "重排服务不可用，已退回 RRF 融合顺序：%s",
                type(error).__name__,
                extra={"request_id": request_id},
            )
            return candidates[:top_n], True, type(error).__name__

        # 重建候选列表：仅覆写重排分，chunk 与溯源字段原样保留（见 item_convert.py）
        return rebuild_rerank_items(candidates, ranked), False, None

    @staticmethod
    def _collapse_parent_chunks(
        items: list[RankedItem],
        top_n: int,
    ) -> list[RankedItem]:
        """归并父子块后再截取最终条数。"""
        return collapse_parent_items(items, top_n)

    def _expansion_phrases(self, question: str) -> list[str]:
        """本次查询会追加的法条用语（未开扩展或未装配扩展器时为空）。"""
        provider = getattr(self.keyword_searcher, "expansion_phrases", None)
        return list(provider(question)) if callable(provider) else []
