from __future__ import annotations

"""批量向 Milvus 导入文档分块和向量的命令行工具。"""

import argparse
import logging
from pathlib import Path

from ..config.config import load_config
from .embedding import LocalEmbeddingClient, build_embedding_text
from .milvus_store import MilvusChunkStore, read_chunks_jsonl
from ..ingestion.pipeline import setup_logging


def ingest_to_milvus(config_path: str, chunks_path: str) -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config(config_path)

    chunk_file = Path(chunks_path) if chunks_path else config.chunking.output_path
    chunks = read_chunks_jsonl(chunk_file)
    logger.info("读取 chunk 数量：%s", len(chunks))
    if not chunks:
        logger.warning("没有可入库的 chunk")
        return

    embedding_client = LocalEmbeddingClient(config.embedding)
    store = MilvusChunkStore(config.milvus)
    store.ensure_collection()

    existing_ids = store.existing_chunk_ids([chunk["chunk_id"] for chunk in chunks])
    pending_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in existing_ids]
    logger.info("已存在 chunk：%s，待新增 chunk：%s", len(existing_ids), len(pending_chunks))
    if not pending_chunks:
        logger.info("没有需要新增的 chunk，跳过向量化和入库")
        return

    batch_size = config.embedding.batch_size
    for start in range(0, len(pending_chunks), batch_size):
        batch = pending_chunks[start : start + batch_size]
        texts = [build_embedding_text(chunk["content"], chunk.get("metadata", {})) for chunk in batch]
        vectors = embedding_client.encode(texts)
        store.insert_chunks(batch, vectors)
        logger.info("入库进度：%s/%s", min(start + batch_size, len(pending_chunks)), len(pending_chunks))

    logger.info("Milvus 入库完成：collection=%s", config.milvus.collection_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="将教育 RAG chunk 向量化并写入 Milvus")
    parser.add_argument("--config", default="configs/crawler.yaml", help="配置文件路径")
    parser.add_argument("--chunks", default="", help="chunk JSONL 文件路径，默认读取配置中的 chunking.output_path")
    args = parser.parse_args()
    ingest_to_milvus(args.config, args.chunks)


if __name__ == "__main__":
    main()
