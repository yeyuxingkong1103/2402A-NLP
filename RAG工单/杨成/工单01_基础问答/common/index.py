"""可持久化的本地文档索引。"""

import json
from pathlib import Path
import re
from common.embedding import LocalEmbedding
from common.models import DocumentChunk, ImageRecord, SearchResult, TableRecord


class DocumentIndex:
    _MIN_RELEVANCE_SCORE = 0.2

    def __init__(self, embedding: LocalEmbedding | None = None) -> None:
        self.embedding = embedding or LocalEmbedding()
        self.chunks: list[DocumentChunk] = []
        self.vectors: list[list[float]] = []

    def add_chunk(self, chunk: DocumentChunk) -> None:
        self.chunks.append(chunk)
        self.vectors.append(self.embedding.embed(chunk.content))

    def add_image(self, image: ImageRecord) -> None:
        """将图像 OCR/描述文本作为普通文档块写入统一索引。"""
        self.add_chunk(DocumentChunk(
            source=image.source,
            page=image.page,
            content=image.content,
            metadata={**image.metadata, "image_id": image.image_id},
        ))

    def add_text(self, source: str, page: int, content: str, metadata: dict | None = None) -> DocumentChunk:
        chunk = DocumentChunk(source=source, page=page, content=content, metadata=metadata or {})
        self.add_chunk(chunk)
        return chunk

    def add_table(self, table: TableRecord) -> None:
        """将表格检索文本作为普通文档块写入统一索引。"""
        self.add_chunk(DocumentChunk(
            source=table.source,
            page=table.page,
            content=table.content,
            metadata={**table.metadata, "table_id": table.table_id},
        ))

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query_tokens = self.embedding.tokenize(query)
        query_terms = set(query_tokens)
        query_vector = self.embedding.embed(query)
        ranked: list[tuple[float, DocumentChunk]] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            content_tokens = self.embedding.tokenize(chunk.content)
            content_counts = {token: content_tokens.count(token) for token in set(content_tokens)}
            coverage = sum(1 for token in query_terms if content_counts.get(token, 0)) / len(query_terms) if query_terms else 0.0
            similarity = self.embedding.cosine_similarity(query_vector, vector)
            score = 0.7 * similarity + 0.3 * coverage
            ranked.append((score, chunk))

        ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
        results: list[SearchResult] = []
        seen: set[str] = set()
        for score, chunk in ranked:
            if score < self._MIN_RELEVANCE_SCORE:
                continue
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            result = SearchResult(
                content=chunk.content,
                source=chunk.source,
                score=score,
                page=chunk.page,
                metadata={**chunk.metadata, "page": chunk.page},
            )
            results.append(result)
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimensions": self.embedding.dimensions,
            "chunks": [
                {"source": chunk.source, "page": chunk.page, "content": chunk.content, "metadata": chunk.metadata}
                for chunk in self.chunks
            ],
            "vectors": self.vectors,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DocumentIndex":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"索引文件无法读取或不是有效 JSON: {path}") from exc

        if not isinstance(payload, dict):
            raise ValueError("损坏的索引: 顶层结构必须是对象")
        required_fields = {"dimensions", "chunks", "vectors"}
        if set(payload) != required_fields:
            raise ValueError("损坏的索引: 顶层字段必须为 dimensions、chunks、vectors")
        dimensions = payload["dimensions"]
        chunks = payload["chunks"]
        vectors = payload["vectors"]
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            raise ValueError("损坏的索引: dimensions 必须是正整数")
        if not isinstance(chunks, list) or not isinstance(vectors, list):
            raise ValueError("损坏的索引: chunks 和 vectors 必须是数组")
        if len(chunks) != len(vectors):
            raise ValueError("损坏的索引: chunks 和 vectors 数量不一致")

        index = cls(LocalEmbedding(dimensions))
        for number, item in enumerate(chunks):
            if not isinstance(item, dict) or set(item) != {"source", "page", "content", "metadata"}:
                raise ValueError(f"损坏的索引: chunks[{number}] 字段不完整或无效")
            if not isinstance(item["source"], str) or not isinstance(item["page"], int) or isinstance(item["page"], bool):
                raise ValueError(f"损坏的索引: chunks[{number}] 的 source 或 page 无效")
            if not isinstance(item["content"], str) or not isinstance(item["metadata"], dict):
                raise ValueError(f"损坏的索引: chunks[{number}] 的 content 或 metadata 无效")
            vector = vectors[number]
            if (
                not isinstance(vector, list)
                or len(vector) != dimensions
                or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in vector)
            ):
                raise ValueError(f"损坏的索引: vectors[{number}] 的向量维度必须为 {dimensions}")
            index.chunks.append(DocumentChunk(
                source=item["source"], page=item["page"], content=item["content"], metadata=item["metadata"]
            ))
            index.vectors.append(vector)
        return index
