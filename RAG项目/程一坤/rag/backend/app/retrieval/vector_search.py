"""法律检索主体：向量召回 → 取正文 → 重排。

三步各司其职：
1. 把问题向量化（BGE-M3，与建索引时同一个模型，维度必须一致）
2. 到 Milvus 召回候选（这一步只保证"相关的不被漏掉"）
3. 回 MySQL 取正文与来源，再用重排模型精排出最相关的几条

数据结构和上下文组装放在 context_builder.py，这里只负责检索流程本身。
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.sql_models import ChunkSummary, Document, DocumentChunk, DocumentVersion, Law, LawVersion
from app.db.version_status import APPROVED
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.context_builder import date_to_timestamp, resolve_current_status
from app.retrieval.assembly import EmbeddingClientProtocol, RerankerProtocol

# 模块边界：过滤器表达式由 filters.py 构造（本模块只接收 filter_expr 字符串），
# 多路结果的 RRF 融合与最终组装在 service.py / assembly.py，本模块只管向量这一路。


class RetrievalError(RuntimeError):
    """检索链路失败（向量化、Milvus 查询或取正文出错）。

    上层据此区分"检索不可用"与"检索没找到"：前者应报错或降级，
    后者才走拒答逻辑 —— 混在一起会让"服务挂了"看起来像"知识库没有"。
    """


class LegalRetriever:
    """法律检索器：把「用户问题」变成「可用于回答的法条片段」。"""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClientProtocol,
        milvus_client: Any,
        collection_name: str,
        session_factory: Any,
        reranker: RerankerProtocol | None = None,
    ) -> None:
        # 向量化客户端：把问题变成 1024 维向量
        self.embedding_client = embedding_client
        # Milvus 客户端：负责相似度检索
        self.milvus_client = milvus_client
        # 集合名：与建索引时使用同一个，否则查不到
        self.collection_name = collection_name
        # 数据库会话工厂：用来取正文（Milvus 里只有检索文本，没有正文）
        self.session_factory = session_factory
        # 重排客户端：可以不传，不传时按召回分数排序
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        *,
        recall_limit: int = 20,
        rerank_top_n: int = 5,
        filter_expr: str | None = None,
    ) -> list[RetrievedArticle]:
        """执行完整检索：召回 → 取正文 → 重排。

        Args:
            question: 用户问题
            recall_limit: 向量召回条数
            rerank_top_n: 重排后保留的条数
            filter_expr: Milvus 过滤表达式（由 filters.build_filter_expression() 生成）
        """
        # 第一步：向量召回。只管"别漏"，排序好坏交给最后的精排
        candidate_keys, recall_scores = self._recall(question, recall_limit, filter_expr)
        # 一条都没召回时直接返回空；是拒答还是换检索策略，由上层决定，这里不擅自兜底
        if not candidate_keys:
            return []

        # 第二步：回 MySQL 取正文与来源信息
        articles = self._load_articles(candidate_keys, recall_scores)
        # 第三步：服务层融合多路结果时可传 0，表示这里只召回、不提前精排
        if rerank_top_n <= 0:
            return sorted(articles, key=lambda item: item.recall_score, reverse=True)
        # 独立使用向量检索器时，仍在向量召回后直接重排
        return self._rerank(question, articles, rerank_top_n)

    def _recall(
        self, question: str, recall_limit: int, filter_expr: str | None = None
    ) -> tuple[list[str], dict[str, float]]:
        """向量召回：返回候选 chunk_key 列表，以及每个候选的相似度分数。"""
        # 调用向量服务把问题编码成向量
        try:
            vectors = self.embedding_client.embed([question])
        except Exception as error:
            # 只保留异常类型，不把可能含密钥的请求细节抛出去
            raise RetrievalError(f"问题向量化失败：{type(error).__name__}") from error

        # 返回数量对不上说明服务端异常；拿空向量查库会静默召回一堆无关内容，必须拦住
        if not vectors:
            raise RetrievalError("问题向量化返回为空")

        # 到 Milvus 做相似度检索；COSINE 必须与建索引时一致，否则分数没有可比性
        try:
            search_params = {"metric_type": "COSINE"}
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[vectors[0]],
                filter=filter_expr or "",
                limit=recall_limit,
                output_fields=["chunk_key"],
                search_params=search_params,
            )
        except Exception as error:
            raise RetrievalError(f"Milvus 召回失败：{type(error).__name__}") from error

        # 解析返回结果。pymilvus 不同版本返回字典或对象，这里统一成同一套字段，
        # 避免上层到处判断类型
        chunk_keys: list[str] = []
        scores: dict[str, float] = {}
        # Milvus 返回 list[list[hit]]（一路查询一个子列表）；本项目只发一路，故取第 0 个
        for hit in (results[0] if results else []):
            # 统一取字段：优先用字典访问，兼容对象式返回
            hit_data = (
                hit if isinstance(hit, dict) else {"id": hit.id, "distance": hit.distance}
            )
            # output_fields 的命中字段挂在 entity 下；没有则退回空字典（避免对 None 取 .get()）
            entity = hit_data.get("entity") or {}
            # chunk_key 是本项目自定义主键；拿不到就退回 Milvus 的 id
            chunk_key = entity.get("chunk_key") or hit_data.get("id")
            if chunk_key:
                chunk_keys.append(str(chunk_key))
                scores[str(chunk_key)] = float(hit_data.get("distance", 0.0))
        return chunk_keys, scores

    def _load_articles(
        self, chunk_keys: Sequence[str], recall_scores: dict[str, float]
    ) -> list[RetrievedArticle]:
        """按 chunk_key 取正文；命中的是子块时，改用它的父块正文。

        理由：子块切得细是为了检索准，但回答需要人读得懂的完整上下文，
        父块（整条法条）通常才是合适的粒度。
        """
        # 两次查询：先取命中的块，再取它们的父块。
        # 分两次是因为父子关系要等命中块取回来才知道，一次 join 拿不全
        with self.session_factory() as session:
            hits = self._fetch_chunks(session, chunk_keys)
            # 收集所有父块 key（去重交给数据库的 in 条件处理）
            parent_keys = [
                row["parent_chunk_key"] for row in hits if row["parent_chunk_key"]
            ]
            parents = self._fetch_chunks(session, parent_keys) if parent_keys else []

        # 父块按 key 建索引，下面替换正文时直接查表，避免循环里再查库
        parent_by_key = {row["chunk_key"]: row for row in parents}

        # 按"正文来源"去重：多条子块可能命中同一个父块，
        # 若不去重，同一条法条会占掉多个引用编号，既浪费上下文位又让模型误判条数
        best_by_source_key: dict[str, RetrievedArticle] = {}
        for row in hits:
            # 有父块就用父块，没有就用命中块自己
            source_row = parent_by_key.get(row["parent_chunk_key"] or "", row)
            # 用来源块的 key 作为去重标识（父块 key 或命中块 key）
            source_key = source_row["chunk_key"]
            recall_score = recall_scores.get(row["chunk_key"], 0.0)

            # 同一来源只保留召回分数最高的那一次命中
            existing = best_by_source_key.get(source_key)
            if existing is not None and existing.recall_score >= recall_score:
                continue

            # 除 chunk_key 外各字段优先取父块（父块是整条，条号与正文更完整）
            best_by_source_key[source_key] = RetrievedArticle(
                # 标识仍用命中的子块 key，父块 key 单独传给服务层统一归并
                chunk_key=row["chunk_key"],
                parent_chunk_key=source_key,
                content=source_row["content"],
                # 条号优先取父块的（父块是整条，条号更完整）
                article_number=source_row["article_number"] or row["article_number"],
                document_title=source_row["document_title"],
                source_url=source_row["source_url"],
                recall_score=recall_score,
                paragraph_number=source_row["paragraph_number"],
                item_number=source_row["item_number"],
                document_type=source_row["document_type"],
                jurisdiction=source_row["jurisdiction"],
                effective_date=source_row["effective_date"],
                expiration_date=source_row["expiration_date"],
                issuing_authority=source_row["issuing_authority"],
                is_current=source_row["is_current"],
                document_id=source_row["document_id"],
                # 摘要跟正文走：只对父块（整条）生成，直接取来源块的摘要
                summary=source_row["summary"],
            )

        # 按召回分数降序返回，保持"最像的在前"这一直觉顺序
        return sorted(
            best_by_source_key.values(), key=lambda article: article.recall_score, reverse=True
        )

    @staticmethod
    def _fetch_chunks(session: Session, chunk_keys: Sequence[str]) -> list[dict[str, Any]]:
        """按 chunk_key 批量取正文，并带上所属文档的标题与来源地址。"""
        # 没有 key 就不查库，避免生成 in () 这种无意义查询
        if not chunk_keys:
            return []

        # 两次 join 是为了拿到「文档标题 + 来源地址」——引用里必须带这两项
        statement = (
            select(
                DocumentChunk.chunk_key,
                DocumentChunk.content,
                DocumentChunk.article_number,
                DocumentChunk.paragraph_number,
                DocumentChunk.item_number,
                DocumentChunk.parent_chunk_key,
                Document.title,
                Document.source_url,
                Document.document_key,
                Law.document_type,
                Law.jurisdiction,
                Law.issuing_authority,
                LawVersion.effective_date,
                LawVersion.expiration_date,
                LawVersion.status,
                # 批次 37：条文一句话摘要（只对 parent 块生成；外连接，未生成时为 None）
                ChunkSummary.summary,
            )
            # 块 → 版本（块的正文属于某个版本）
            .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
            # 版本 → 文档（标题与来源挂在文档上）
            .join(Document, DocumentVersion.document_id == Document.id)
            # 阶段6（6.3②）检索侧兜底：只取已审核发布的版本正文。
            # 历史遗留向量可能把未审核内容带进召回，这里在取正文时再次把关，
            # 未审核版本的 chunk 会因取不到正文而被静默丢弃
            .where(
                DocumentChunk.chunk_key.in_(list(chunk_keys)),
                DocumentVersion.version_status == APPROVED,
            )
            # 历史数据可能尚未补齐法规元数据，因此使用外连接，避免正文被过滤掉。
            .outerjoin(LawVersion, LawVersion.document_version_id == DocumentVersion.id)
            .outerjoin(Law, Law.id == LawVersion.law_id)
            # 批次 37：摘要表按 chunk_key 对齐（主键即 chunk_key），未生成摘要不影响取正文
            .outerjoin(ChunkSummary, ChunkSummary.chunk_key == DocumentChunk.chunk_key)
        )

        # 执行查询并把结果行转成普通字典，方便上层按字段名取用
        rows = session.execute(statement).all()
        return [
            {
                "chunk_key": row.chunk_key,
                "content": row.content,
                "article_number": row.article_number,
                "paragraph_number": row.paragraph_number,
                "item_number": row.item_number,
                "parent_chunk_key": row.parent_chunk_key,
                "document_title": row.title,
                "source_url": row.source_url,
                "document_id": row.document_key,
                "document_type": row.document_type,
                "jurisdiction": row.jurisdiction,
                "issuing_authority": row.issuing_authority,
                "effective_date": date_to_timestamp(row.effective_date, unknown=0),
                "expiration_date": date_to_timestamp(row.expiration_date),
                "is_current": resolve_current_status(row.status, row.expiration_date),
                "summary": row.summary,
            }
            for row in rows
        ]

    def _rerank(
        self,
        question: str,
        articles: Sequence[RetrievedArticle],
        rerank_top_n: int,
    ) -> list[RetrievedArticle]:
        """用重排模型精排；没有重排客户端时按召回分数取前 N 条。"""
        # 没有候选就没有可排的，直接返回空（上层按空结果走拒答）
        if not articles:
            return []

        # 未接重排服务时按召回分数排序，保证链路仍能跑通
        if self.reranker is None:
            ordered = sorted(articles, key=lambda item: item.recall_score, reverse=True)
            return list(ordered[:rerank_top_n])

        try:
            # 把候选正文送去重排，返回按相关度降序的下标与分数
            ranked = self.reranker.rerank(
                question, [article.content for article in articles], top_n=rerank_top_n
            )
        except Exception:
            # 重排是"锦上添花"的一步：它失败不应让整个回答失败，
            # 退回召回顺序，保证用户至少能拿到可用的法源
            ordered = sorted(articles, key=lambda item: item.recall_score, reverse=True)
            return list(ordered[:rerank_top_n])

        # 按重排结果顺序重建列表，并把重排分数一起带上（便于观察与调参）
        return [
            replace(
                articles[candidate.index],
                rerank_score=float(candidate.score),
            )
            for candidate in ranked
        ]
