import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFilters:
    # 检索过滤条件只保存案件时间和关系类型，避免携带用户问题全文。
    query_date: Any | None = None
    relationship_type: str = "general"
    materials: list[Any] = field(default_factory=list)


@dataclass
class RetrievedDocument:
    # 检索片段承载最小可引用信息，正文只在最终上下文中短摘录使用。
    id: str
    material_id: str
    version_id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        # 兼容早期 dict 风格测试和调用方读取方式。
        return getattr(self, key)


def as_retrieved_document(item: RetrievedDocument | dict[str, Any], default_source: str = "unknown") -> RetrievedDocument:
    # 已是标准对象时直接返回，避免重复转换丢失元数据。
    if isinstance(item, RetrievedDocument):
        return item
    # 兼容测试和早期调用方的 dict 结构，缺省字段使用 id 兜底。
    item_id = str(item.get("id", item.get("material_id", "")))
    metadata = dict(item.get("metadata", {}))
    return RetrievedDocument(
        id=item_id,
        material_id=str(item.get("material_id", item_id)),
        version_id=str(item.get("version_id", item_id)),
        text=str(item.get("text", "")),
        score=float(item.get("score", 0.0)),
        source=str(item.get("source", default_source)),
        metadata=metadata,
    )


def document_dedup_key(document: RetrievedDocument) -> tuple[str, str, str, str, str]:
    # 去重键严格按材料、版本、条、款和正文哈希组合，避免跨版本误合并。
    article = str(document.metadata.get("article", ""))
    paragraph = str(document.metadata.get("paragraph", ""))
    text_hash = str(document.metadata.get("text_hash") or hashlib.sha256(document.text.encode("utf-8")).hexdigest())
    return (document.material_id, document.version_id, article, paragraph, text_hash)


def merge_retrieved_documents(
    vector_results: list[RetrievedDocument | dict[str, Any]],
    bm25_results: list[RetrievedDocument | dict[str, Any]],
) -> list[RetrievedDocument]:
    # 每路最多取 Top 20，符合混合检索治理要求。
    normalized = [as_retrieved_document(item, "dense") for item in vector_results[:20]]
    normalized.extend(as_retrieved_document(item, "bm25") for item in bm25_results[:20])
    merged: list[RetrievedDocument] = []
    seen: dict[tuple[str, str, str, str, str], RetrievedDocument] = {}
    for document in normalized:
        # 每条候选都记录来源分数，便于排查融合效果但不记录正文。
        key = document_dedup_key(document)
        if key not in seen:
            document.metadata = _metadata_with_source_score(document, document.metadata)
            seen[key] = document
            merged.append(document)
            continue
        _merge_source_metadata(seen[key], document)
    logger.info(
        "混合检索结果合并完成",
        extra={"vector_count": min(len(vector_results), 20), "bm25_count": min(len(bm25_results), 20), "merged_count": len(merged)},
    )
    return merged


def _metadata_with_source_score(document: RetrievedDocument, metadata: dict[str, Any]) -> dict[str, Any]:
    # 复制元数据，避免修改调用方传入对象中的共享字典。
    updated = dict(metadata)
    # 记录单路来源列表，后续去重命中时追加另一来源。
    updated["sources"] = [document.source]
    if document.source == "dense":
        updated["dense_score"] = document.score
    elif document.source == "bm25":
        updated["bm25_score"] = document.score
    else:
        updated[f"{document.source}_score"] = document.score
    return updated


def _merge_source_metadata(existing: RetrievedDocument, duplicate: RetrievedDocument) -> None:
    # 来源列表保持发现顺序，避免重复写入同一来源。
    sources = existing.metadata.setdefault("sources", [])
    if duplicate.source not in sources:
        sources.append(duplicate.source)
    # 分数按来源独立保存，不把 BM25 分数和向量分数混为一个排序值。
    if duplicate.source == "dense":
        existing.metadata["dense_score"] = duplicate.score
    elif duplicate.source == "bm25":
        existing.metadata["bm25_score"] = duplicate.score
    else:
        existing.metadata[f"{duplicate.source}_score"] = duplicate.score
