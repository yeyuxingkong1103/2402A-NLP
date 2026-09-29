from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.law_status import EXPIRED_STATUSES
from app.db.sql_models import DocumentChunk, DocumentVersion, Law, LawVersion
from app.db.version_status import APPROVED


class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class VectorStore(Protocol):
    def existing_chunk_keys(self, chunk_keys: Sequence[str]) -> set[str]:
        # 查询候选 chunk 中已经写入向量库的 key，供失败重试时跳过。
        ...

    def upsert(self, rows: list[dict]) -> None:
        ...

    def get_all_chunk_keys(self) -> set[str]:
        # 查询向量库中的所有 chunk_key
        ...

    def delete_by_chunk_keys(self, chunk_keys: Sequence[str]) -> int:
        # 根据 chunk_key 批量删除向量，返回删除数量
        ...


@dataclass(frozen=True)
class VectorIndexResult:
    status: str
    indexed_versions: int
    indexed_chunks: int


def index_awaiting_embeddings(
    session: Session,
    *,
    embedding_client: EmbeddingClient,
    vector_store: VectorStore,
    batch_size: int,
    embedding_batch_size: int = 32,
) -> VectorIndexResult:
    # 阶段6（6.3①）：只索引已审核发布的版本——未审核内容根本不进 Milvus；
    # 这比"检索时过滤"更彻底，也不会出现召回后取不到正文的情况
    versions = session.scalars(
        select(DocumentVersion)
        .where(
            DocumentVersion.processing_status == "awaiting_embedding",
            DocumentVersion.version_status == APPROVED,
        )
        .order_by(DocumentVersion.id)
        .limit(batch_size)
    ).all()
    if not versions:
        return VectorIndexResult(status="empty", indexed_versions=0, indexed_chunks=0)

    version_ids = [version.id for version in versions]
    chunks = session.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_version_id.in_(version_ids))
        .order_by(DocumentChunk.document_version_id, DocumentChunk.sequence)
    ).all()
    if not chunks:
        return VectorIndexResult(status="empty", indexed_versions=0, indexed_chunks=0)

    if embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be greater than zero")

    # 查询当前版本已经写入向量库的 chunk，避免失败重试时重复请求 Embedding。
    existing_chunk_keys = vector_store.existing_chunk_keys(
        [chunk.chunk_key for chunk in chunks]
    )
    # 仅保留尚未写入向量库的 chunk，确保重试只处理未完成数据。
    pending_chunks = [
        chunk for chunk in chunks if chunk.chunk_key not in existing_chunk_keys
    ]
    # 没有待处理 chunk 时，直接将版本标记为 indexed。
    if not pending_chunks:
        for version in versions:
            version.processing_status = "indexed"
        session.commit()
        return VectorIndexResult(
            status="indexed",
            indexed_versions=len(versions),
            indexed_chunks=0,
        )

    # 按小批次调用 Embedding，避免单次请求过大导致超时。
    indexed_chunks = 0
    for chunk_start in range(0, len(pending_chunks), embedding_batch_size):
        chunk_batch = pending_chunks[chunk_start : chunk_start + embedding_batch_size]
        vectors = embedding_client.embed([chunk.retrieval_text for chunk in chunk_batch])

        # 查询时效和法域信息（从三层表）
        chunk_metadata = _fetch_chunk_metadata(session, chunk_batch)
        # 批次 37：批量取条文摘要（chunk_summaries 表）；未生成的写 None（Milvus 可空字段），
        # 摘要是离线任务，不阻塞索引——重新 summarize 后重索引即可补上。
        summaries = _fetch_summaries(session, [chunk.chunk_key for chunk in chunk_batch])

        rows = [
            {
                "chunk_key": chunk.chunk_key,
                "document_version_id": chunk.document_version_id,
                "chunk_type": chunk.chunk_type,
                "article_number": chunk.article_number,
                "retrieval_text": chunk.retrieval_text,
                # 时效和法域字段
                "law_name": chunk_metadata.get(chunk.chunk_key, {}).get("law_name", "未知法规"),
                "document_type": chunk_metadata.get(chunk.chunk_key, {}).get("document_type", "待确认"),
                "jurisdiction": chunk_metadata.get(chunk.chunk_key, {}).get("jurisdiction", "中国大陆"),
                "authority_level": chunk_metadata.get(chunk.chunk_key, {}).get("authority_level", 99),
                "effective_date": chunk_metadata.get(chunk.chunk_key, {}).get("effective_date", 0),
                "expiration_date": chunk_metadata.get(chunk.chunk_key, {}).get("expiration_date"),
                "is_current": chunk_metadata.get(chunk.chunk_key, {}).get("is_current", True),
                "article_path": chunk_metadata.get(chunk.chunk_key, {}).get("article_path", ""),
                # 批次 37：一句话摘要（引用卡片展示与父块检索补充用）
                "summary": summaries.get(chunk.chunk_key),
                "vector": vector,
            }
            for chunk, vector in zip(chunk_batch, vectors, strict=True)
        ]
        vector_store.upsert(rows)
        indexed_chunks += len(rows)
    for version in versions:
        version.processing_status = "indexed"
    session.commit()
    return VectorIndexResult(
        status="indexed",
        indexed_versions=len(versions),
        indexed_chunks=indexed_chunks,
    )


@dataclass(frozen=True)
class PruneOrphanVectorsResult:
    deleted_count: int
    deleted_keys_sample: list[str]


def prune_orphan_vectors(
    session: Session,
    vector_store: VectorStore,
) -> PruneOrphanVectorsResult:
    """清理孤儿向量：删除 Milvus 中存在但 MySQL document_chunks 中不存在的向量。

    用途：强制重建导入（--replace-existing）后，旧 chunk 被删除但 Milvus 向量未清理，
    导致检索命中已不存在的 chunk_key。
    """
    # 查询 MySQL 中所有有效的 chunk_key
    mysql_chunk_keys = set(
        session.scalars(select(DocumentChunk.chunk_key)).all()
    )

    # 查询 Milvus 中所有的 chunk_key
    milvus_chunk_keys = vector_store.get_all_chunk_keys()

    # 找出孤儿向量（在 Milvus 但不在 MySQL）
    orphan_keys = milvus_chunk_keys - mysql_chunk_keys

    if not orphan_keys:
        return PruneOrphanVectorsResult(deleted_count=0, deleted_keys_sample=[])

    # 批量删除孤儿向量
    deleted_count = vector_store.delete_by_chunk_keys(list(orphan_keys))

    # 返回删除数量和样例（最多10个）
    return PruneOrphanVectorsResult(
        deleted_count=deleted_count,
        deleted_keys_sample=list(orphan_keys)[:10],
    )


def _fetch_summaries(session: Session, chunk_keys: list[str]) -> dict[str, str]:
    """批量取条文摘要（chunk_summaries 表，批次 37 新增）。

    Returns:
        {chunk_key: summary}；无摘要的 chunk 不出现在结果里，
        调用方用 .get() 落 None —— 摘要缺失不阻塞索引。
    """
    if not chunk_keys:
        return {}
    from app.db.sql_models import ChunkSummary

    rows = session.execute(
        select(ChunkSummary.chunk_key, ChunkSummary.summary).where(
            ChunkSummary.chunk_key.in_(chunk_keys)
        )
    ).all()
    return {chunk_key: summary for chunk_key, summary in rows}


def _fetch_chunk_metadata(session: Session, chunks: list[DocumentChunk]) -> dict[str, dict]:
    """查询 chunk 的时效和法域元数据（从三层表）。

    Returns:
        {chunk_key: {law_name, document_type, jurisdiction, authority_level,
                     effective_date, expiration_date, is_current, article_path}}
    """
    from datetime import datetime

    version_ids = list({chunk.document_version_id for chunk in chunks})

    # 联表查询：DocumentChunk -> LawVersion -> Law
    results = session.execute(
        select(
            DocumentChunk.chunk_key,
            Law.name,
            Law.document_type,
            Law.jurisdiction,
            Law.authority_level,
            LawVersion.effective_date,
            LawVersion.expiration_date,
            LawVersion.status,
            DocumentChunk.article_number,
            DocumentChunk.paragraph_number,
            DocumentChunk.item_number,
        )
        .join(LawVersion, LawVersion.document_version_id == DocumentChunk.document_version_id)
        .join(Law, Law.id == LawVersion.law_id)
        .where(DocumentChunk.document_version_id.in_(version_ids))
    ).all()

    metadata = {}
    for row in results:
        chunk_key, law_name, doc_type, jurisdiction, auth_level, eff_date, exp_date, status, art_no, para_no, item_no = row

        # 构建 article_path
        # 条/款/项在分块里可能是中文形式（"第四十条"），而 articles 表统一存阿拉伯数字；
        # 这里必须做同样的规范化，否则向量库与关系库的条文路径对不上，
        # 按条号过滤时会查不到东西。
        from app.ingest.chinese_number import to_arabic_number

        normalized_article = to_arabic_number(art_no)
        normalized_paragraph = to_arabic_number(para_no)
        normalized_item = to_arabic_number(item_no)

        article_path = normalized_article or ""
        if normalized_paragraph:
            article_path += f"-{normalized_paragraph}"
        if normalized_item:
            article_path += f"-{normalized_item}"

        # 转换日期为 Unix timestamp。
        # 0 表示"生效日期未知（待人工补录）"，不是真实的 1970-01-01；
        # 检索侧做时效过滤时必须显式排除 0，否则未知日期会被当成"一直有效"。
        eff_timestamp = int(datetime.combine(eff_date, datetime.min.time()).timestamp()) if eff_date else 0
        exp_timestamp = int(datetime.combine(exp_date, datetime.min.time()).timestamp()) if exp_date else None

        # 是否现行有效：只有页面/数据明确写了已废止 / 已失效 才算失效。
        # 失效取值取自 app/db/law_status.py（唯一定义处）——历史上这里手抄中文元组，
        # 写进库的英文取值静默漏过该判断，正是本次统一的根因。
        # 状态未知时不判失效（True）——缺数据不能成为"静默隐藏条文"的理由，
        # 否则大部分待补录状态的法规会在时效检索里凭空消失；
        # 不适用（案例材料）**不参与时效判断**，同样写 True
        # （Milvus 的 is_current 是非空 BOOL，写不了 None）。
        expired_by_status = status in EXPIRED_STATUSES
        is_current = (not expired_by_status) and (
            exp_date is None or exp_date > datetime.now().date()
        )

        metadata[chunk_key] = {
            "law_name": law_name,
            "document_type": doc_type,
            "jurisdiction": jurisdiction,
            "authority_level": auth_level,
            "effective_date": eff_timestamp,
            "expiration_date": exp_timestamp,
            "is_current": is_current,
            "article_path": article_path,
        }

    return metadata
