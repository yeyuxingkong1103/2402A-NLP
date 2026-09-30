from __future__ import annotations

"""RAG 检索编排层，组合向量检索、BM25、RRF 融合和重排序。"""

import json
from dataclasses import dataclass
from typing import Any

from ..config.config import AppConfig
from .embedding import LocalEmbeddingClient
from .filtering import matches_metadata
from .hybrid_search import BM25Index, KeywordResult
from .milvus_store import MilvusChunkStore
from .reranker import CrossEncoderReranker


@dataclass(frozen=True)
class RetrievedChunk:
    """检索结果的统一内部表示。"""
    score: float
    chunk_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]


class MilvusRetriever:
    """组合 Milvus、BM25 和可选重排序器的混合检索器。"""
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedding_client = LocalEmbeddingClient(config.embedding)
        self.store = MilvusChunkStore(config.milvus)
        self.store.client.load_collection(config.milvus.collection_name)
        self.keyword_index = BM25Index.from_jsonl(config.chunking.output_path) if config.hybrid_search.enabled else None
        self.reranker = CrossEncoderReranker(config.rerank)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        final_top_k = top_k or self.config.qa.top_k
        vector_top_k = self._candidate_limit(final_top_k)
        vector_results = self._vector_search(query, vector_top_k, filters)
        vector_results = [chunk for chunk in vector_results if matches_metadata(chunk.metadata, filters)]
        if not self.keyword_index:
            return self.reranker.rerank(query, vector_results, final_top_k)

        keyword_results = self.keyword_index.search(
            query,
            self.config.hybrid_search.keyword_top_k,
            filters=filters,
        )
        if not keyword_results:
            return self.reranker.rerank(query, vector_results, final_top_k)
        fused_results = self._fuse_results(vector_results, keyword_results, vector_top_k)
        return self.reranker.rerank(query, fused_results, final_top_k)

    def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        vector = self.embedding_client.encode([query])[0]
        results = self.store.client.search(
            collection_name=self.config.milvus.collection_name,
            data=[vector],
            anns_field="vector",
            limit=top_k,
            output_fields=["id", "chunk_index", "content", "metadata_json"],
            filter=self.store.build_filter(filters),
            search_params={
                "metric_type": self.config.milvus.metric_type,
                "params": self.config.milvus.search_params,
            },
        )[0]
        return [self._to_chunk(result) for result in results]

    def _candidate_limit(self, top_k: int) -> int:
        return max(top_k, top_k * max(1, self.config.hybrid_search.candidate_multiplier))

    def _fuse_results(
        self,
        vector_results: list[RetrievedChunk],
        keyword_results: list[KeywordResult],
        top_k: int,
    ) -> list[RetrievedChunk]:
        chunks_by_id: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in vector_results}
        for result in keyword_results:
            chunks_by_id.setdefault(
                result.chunk_id,
                RetrievedChunk(
                    score=result.score,
                    chunk_id=result.chunk_id,
                    chunk_index=result.chunk_index,
                    content=result.content,
                    metadata=result.metadata,
                ),
            )

        fused_scores: dict[str, float] = {}
        rrf_k = self.config.hybrid_search.rrf_k
        self._add_rrf_scores(fused_scores, [chunk.chunk_id for chunk in vector_results], self.config.hybrid_search.vector_weight, rrf_k)
        self._add_rrf_scores(
            fused_scores,
            [result.chunk_id for result in keyword_results],
            self.config.hybrid_search.keyword_weight,
            rrf_k,
        )

        ranked_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
        return [
            RetrievedChunk(
                score=fused_scores[chunk_id],
                chunk_id=chunks_by_id[chunk_id].chunk_id,
                chunk_index=chunks_by_id[chunk_id].chunk_index,
                content=chunks_by_id[chunk_id].content,
                metadata=chunks_by_id[chunk_id].metadata,
            )
            for chunk_id in ranked_ids[:top_k]
        ]

    @staticmethod
    def _add_rrf_scores(scores: dict[str, float], chunk_ids: list[str], weight: float, rrf_k: int) -> None:
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)

    @staticmethod
    def _to_chunk(result: dict[str, Any]) -> RetrievedChunk:
        entity = result.get("entity", {})
        return RetrievedChunk(
            score=float(result.get("distance", 0.0)),
            chunk_id=str(entity.get("id", "")),
            chunk_index=int(entity.get("chunk_index", 0)),
            content=str(entity.get("content", "")),
            metadata=_parse_metadata(str(entity.get("metadata_json", "{}"))),
        )


def build_teacher_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    max_context_chars: int,
    memory_context: str = "",
) -> str:
    context_parts: list[str] = []
    used_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        citation = _citation_label(index, chunk)
        content = chunk.content.strip()
        part = f"【资料{index}｜{citation}】\n{content}"
        if used_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)
        used_chars += len(part)

    context = "\n\n".join(context_parts) if context_parts else "未检索到相关资料。"
    memory_section = memory_context.strip() or "无可用历史记忆。"
    return f"""你是一个面向九年级语文教师的教学 RAG 助手。
请严格依据【检索资料】回答教师问题。
如果资料不足，请明确说明“当前资料不足，无法确定”，不要编造课文、考试要求或政策。
可以在依据资料的基础上做教师可用的归纳，但必须保持来源可追溯。

【近期对话与长期记忆】
{memory_section}

【教师问题】
{question}

【检索资料】
{context}

【回答要求】
1. 先给出直接答案。
2. 用条目化方式组织内容，方便教师备课或命题参考。
3. 如果涉及课标、教学案例、试题，请说明依据来自哪类资料。
4. 最后列出“引用来源”，格式为：资料编号 + 资料类型 + 章节/主题。
5. 不要输出没有依据的内容。

请生成回答："""


def build_citations(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """把内部检索结果转换为前端可展示的引用信息。"""
    citations = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        citations.append(
            {
                "index": index,
                "score": chunk.score,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "document_type": metadata.get("document_type", ""),
                "title": metadata.get("title", ""),
                "section": metadata.get("section", ""),
                "topic": metadata.get("topic", ""),
                "stage": metadata.get("stage", ""),
                "source_url": metadata.get("source_url", ""),
                "source_file": metadata.get("source_file", ""),
            }
        )
    return citations


def _citation_label(index: int, chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata
    labels = [
        metadata.get("document_type", ""),
        metadata.get("volume", ""),
        metadata.get("section", "") or metadata.get("topic", ""),
    ]
    label = " / ".join(item for item in labels if item)
    return label or f"chunk {chunk.chunk_index}"


def _parse_metadata(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
