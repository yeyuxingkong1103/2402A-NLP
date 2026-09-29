"""RankedItem 与召回结果的转换纯函数（从 retrieval/service.py 抽出的独立块）。

service.retrieve 主链路里三段最大的内联代码都是"字段逐项搬运"：
- 向量召回结果 → RankedItem（进入融合）
- 重排结果 → 重建 RankedItem（覆写重排分）
- 候选 → 重排输入文本（锚点拼接）

它们不依赖任何服务状态，抽成纯函数后主链路只剩"编排"，
且转换规则可被单测直接覆盖。逻辑与抽出前逐行一致（AST 比对背书）。
"""

from app.retrieval.fusion import RankedItem


def vector_articles_to_ranked_items(vector_articles: list) -> list[RankedItem]:
    """把向量召回的 Article 列表转成 RankedItem 列表（keyword 分数留空）。

    召回分同时写入 score / vector_score / recall_score 三处：
    score 供 RRF 排名使用，后两者供统计与展示追溯。
    parent_chunk_key 缺失时退回 chunk_key 自身（父块即本块）。
    """
    return [
        RankedItem(
            chunk_key=art.chunk_key,
            # score 即召回分：RRF 只认排名，score 在融合阶段参与排序
            score=art.recall_score,
            source="vector",
            # 单路来源固定为元组，融合去重时按并集合并
            sources=("vector",),
            parent_chunk_key=getattr(art, "parent_chunk_key", None) or art.chunk_key,
            content=art.content,
            document_title=art.document_title,
            article_number=art.article_number,
            source_url=art.source_url,
            vector_score=art.recall_score,
            recall_score=art.recall_score,
            rerank_score=art.rerank_score,
            paragraph_number=art.paragraph_number,
            item_number=art.item_number,
            document_type=art.document_type,
            jurisdiction=art.jurisdiction,
            effective_date=art.effective_date,
            expiration_date=art.expiration_date,
            issuing_authority=art.issuing_authority,
            is_current=art.is_current,
            document_id=art.document_id,
        )
        for art in vector_articles
    ]


def rebuild_rerank_items(
    candidates: list[RankedItem],
    ranked: list,
) -> list[RankedItem]:
    """用重排结果重建候选列表：chunk 与溯源字段原样保留，仅覆写排序分。

    score 与 rerank_score 都写成重排分——score 是下游排序唯一依据，
    必须随重排结果变化；其余字段从原候选照抄，保证引用信息不丢失。
    ranked 元素的 index 指向 candidates 下标（重排器契约）。
    """
    return [
        RankedItem(
            # item.index 是重排器返回的原候选下标（契约见 reranker 实现）
            chunk_key=candidates[item.index].chunk_key,
            # 重排分直接作为排序分：下游截取只看 score，不重复比较两套分数
            score=float(item.score),
            source="rerank",
            content=candidates[item.index].content,
            document_title=candidates[item.index].document_title,
            article_number=candidates[item.index].article_number,
            source_url=candidates[item.index].source_url,
            parent_chunk_key=candidates[item.index].parent_chunk_key,
            sources=candidates[item.index].sources,
            vector_score=candidates[item.index].vector_score,
            keyword_score=candidates[item.index].keyword_score,
            recall_score=candidates[item.index].recall_score,
            fusion_score=candidates[item.index].fusion_score,
            rerank_score=float(item.score),
            paragraph_number=candidates[item.index].paragraph_number,
            item_number=candidates[item.index].item_number,
            document_type=candidates[item.index].document_type,
            jurisdiction=candidates[item.index].jurisdiction,
            effective_date=candidates[item.index].effective_date,
            expiration_date=candidates[item.index].expiration_date,
            issuing_authority=candidates[item.index].issuing_authority,
            is_current=candidates[item.index].is_current,
            document_id=candidates[item.index].document_id,
        )
        for item in ranked
    ]


def _build_rerank_text(candidate: RankedItem) -> str:
    """批次 12-B：重排输入加法规名/条号锚点。

    纯正文时重排模型只能靠语义猜场景（实测列举式条文被语义稀释，
    golden 46 条重排分仅 0.07~0.13）。锚点格式：「《法规名》条号 正文」；
    案例类候选无条号时只带法规名。
    """
    anchor = ""
    # 法规名取中文书名号包裹；条号直接拼接在书名号后，与法条引用习惯一致
    if candidate.document_title:
        anchor = f"《{candidate.document_title}》"
    # 无条号（案例类候选）时只带法规名；两者皆无则退回纯正文
    if candidate.article_number:
        anchor = f"{anchor}{candidate.article_number}"
    content = candidate.content or ""
    return f"{anchor} {content}" if anchor else content


def articles_to_ranked_items(articles: list) -> list[RankedItem]:
    """把已召回文章转换为可统一重排的候选载体。"""
    return [
        RankedItem(
            chunk_key=article.chunk_key,
            score=article.recall_score,
            source="retry_merge",
            content=article.content,
            document_title=article.document_title,
            article_number=article.article_number,
            source_url=article.source_url,
            vector_score=article.vector_score,
            recall_score=article.recall_score,
            paragraph_number=article.paragraph_number,
            item_number=article.item_number,
            document_type=article.document_type,
            jurisdiction=article.jurisdiction,
            effective_date=article.effective_date,
            expiration_date=article.expiration_date,
            issuing_authority=article.issuing_authority,
            is_current=article.is_current,
            document_id=article.document_id,
        )
        for article in articles
    ]


def ranked_items_to_articles(items: list[RankedItem]) -> list: 
    """把统一重排后的候选恢复为问答层使用的文章对象。"""
    from app.retrieval.context_builder import RetrievedArticle

    return [
        RetrievedArticle(
            chunk_key=item.chunk_key,
            content=item.content or "",
            article_number=item.article_number,
            document_title=item.document_title or "",
            source_url=item.source_url or "",
            recall_score=item.recall_score or item.vector_score or 0.0,
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
        for item in items
    ]
