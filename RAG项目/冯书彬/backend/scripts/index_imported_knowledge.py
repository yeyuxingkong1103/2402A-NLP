import argparse
import hashlib
import json
from typing import Any

from sqlalchemy import create_engine, text

from backend.app.core.config import settings
from backend.app.database.milvus import create_milvus_store
from backend.app.embeddings.embedding_factory import get_embedding_client


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_chunks(engine) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT
            dc.id AS chunk_id,
            dc.chunk_index,
            dc.content,
            d.id AS document_id,
            d.title,
            km.id AS material_id,
            km.snapshot_id,
            km.source_url,
            km.publisher,
            km.material_type,
            km.status,
            km.searchable,
            km.effective_from,
            km.created_at,
            km.updated_at
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        JOIN knowledge_materials km ON km.id = d.material_id
        WHERE km.status = 'published' AND km.searchable = true
        ORDER BY km.id, d.id, dc.chunk_index
        """
    )
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(query).mappings().all()]


def build_row(item: dict[str, Any], vector: list[float]) -> dict[str, Any]:
    content = str(item["content"])
    metadata = {
        "source_url": item["source_url"],
        "publisher": item["publisher"],
        "material_type": item["material_type"],
        "status": item["status"],
        "searchable": bool(item["searchable"]),
        "effective_from": item["effective_from"],
        "snapshot_id": item["snapshot_id"],
        "created_at": item["created_at"].isoformat() if item["created_at"] else None,
        "updated_at": item["updated_at"].isoformat() if item["updated_at"] else None,
        "article": None,
        "paragraph": str(item["chunk_index"]),
        "relationship_types": ["general"],
        "text_hash": content_hash(content),
        "title": item["title"],
    }
    return {
        "id": f"{item['material_id']}:{item['document_id']}:{item['chunk_index']}",
        "material_id": item["material_id"],
        "version_id": item["document_id"],
        "relationship_type": "general",
        "text": content[:4000],
        "metadata": metadata,
        "vector": vector,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="从 MySQL 已导入材料批量生成向量并写入 Milvus")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须大于 0")

    engine = create_engine(settings.DATABASE_URL, future=True)
    chunks = load_chunks(engine)
    print(f"待索引 chunks：{len(chunks)}")
    if not chunks:
        return

    embedding_client = get_embedding_client()
    vector_store = create_milvus_store(settings)
    vector_store.ensure_collection(settings.MILVUS_VECTOR_DIMENSION)
    written = 0
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        vectors = embedding_client.embed_texts([str(item["content"])[:4000] for item in batch])
        rows = [build_row(item, vector) for item, vector in zip(batch, vectors, strict=True)]
        written += vector_store.upsert(rows)
        print(f"Milvus 索引进度：{min(start + len(batch), len(chunks))}/{len(chunks)}")
    print(f"Milvus 写入完成：{written}")


if __name__ == "__main__":
    main()
