import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .cleaning import DocumentCleaner
from .chunking import DocumentChunker
from .config import IngestionConfig
from .embedding import BgeEmbedder
from .milvus_store import MilvusKnowledgeStore
from .pipeline import PdfIngestionPipeline


# 统一配置日志格式，方便在命令行里观察入库进度。
def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# 这个函数是“文档进入向量库”的核心调度函数。
# 它把本地 chunk 文件依次经过：读取 -> 去重 -> 生成向量 -> 组装字段 -> 写入 Milvus。
def embed_chunks(config: IngestionConfig, document_id: str | None = None) -> None:
    # 找到 chunks_dir 下所有已经切好的 JSON 文件。
    chunk_files = sorted(config.chunks_dir.glob("*.json"))
    # 如果传了 document_id，就只处理这一篇文档，而不是整个目录。
    if document_id is not None:
        chunk_files = [path for path in chunk_files if path.stem == document_id]
    # 没找到 chunk 文件，说明前面的解析/切块步骤还没完成。
    if not chunk_files:
        raise FileNotFoundError(f"No chunk JSON files found in {config.chunks_dir}")

    # 加载 embedding 模型，后面用它把每个 chunk 的 text 转成向量。
    embedder = BgeEmbedder(config.embedding_model_name, config.embedding_batch_size)
    # 模型输出维度必须和配置一致，否则后面无法写入固定维度的 Milvus 字段。
    if embedder.dimension != config.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: model={embedder.dimension}, configured={config.embedding_dim}"
        )

    # 创建 Milvus 存储对象。
    # 这里会连接 Milvus，并创建或复用知识库 collection。
    store = MilvusKnowledgeStore(
        host=config.milvus_host,
        port=config.milvus_port,
        collection_name=config.milvus_collection_knowledge,
        dimension=embedder.dimension,
    )

    # manifest 用来记录每个文件处理了多少 chunk，最后会保存成 manifest.json。
    manifest: list[dict[str, object]] = []
    inserted_total = 0
    try:
        # 一个 chunk 文件通常对应一篇文档，逐个处理。
        for chunk_file in chunk_files:
            # 读取切块后的 JSON 文档。
            document = json.loads(chunk_file.read_text(encoding="utf-8"))
            chunks = document.get("chunks", [])

            # 先拿到当前文件所有 chunk_id。
            chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
            # 查 Milvus 里哪些 chunk_id 已经存在。
            existing = store.has_chunk_ids(chunk_ids)
            # 只保留还没有入库的 chunk，避免重复写入。
            pending = [chunk for chunk in chunks if str(chunk["chunk_id"]) not in existing]

            # 取出待入库 chunk 的正文，准备批量生成向量。
            texts = [str(chunk["text"]) for chunk in pending]
            # 一次性把这些正文转换成向量。
            vectors = embedder.encode(texts)

            # rows 是 MilvusKnowledgeStore.insert_chunks() 需要的统一数据格式。
            rows = []
            # zip 会把每个待入库 chunk 和它对应的向量配对。
            for index, (chunk, vector) in enumerate(zip(pending, vectors), start=1):
                rows.append(
                    {
                        # 下面这些是文本片段的元数据和正文。
                        "chunk_id": str(chunk["chunk_id"]),
                        "document_id": str(document.get("document_id", "")),
                        "source_file": Path(str(document.get("source_path", ""))).name,
                        "page_start": int(chunk.get("page_start", 0)),
                        "page_end": int(chunk.get("page_end", 0)),
                        # 如果有页码元数据就使用它，否则使用当前循环序号。
                        "chunk_index": int(chunk.get("metadata", {}).get("page", index)),
                        "chunk_type": "text",
                        "text": str(chunk["text"]),
                        # vector 就是刚才由 embedding 模型生成的数字向量。
                        "vector": vector,
                    }
                )

            # 把当前文件整理好的 rows 批量写入 Milvus。
            inserted = store.insert_chunks(rows)
            inserted_total += inserted

            # 记录这篇文档的入库统计信息。
            manifest.append(
                {
                    "source": chunk_file.name,
                    "total_chunks": len(chunks),
                    "skipped_existing": len(existing),
                    "inserted": inserted,
                }
            )
            # 打印日志，方便知道当前文件到底插入了多少、跳过了多少。
            logging.getLogger(__name__).info(
                "Embedded %s: inserted=%d skipped=%d", chunk_file.name, inserted, len(existing)
            )
    finally:
        # 不管中途成功还是报错，都要关闭 Milvus 连接。
        store.close()

    # 把本次入库信息写到 vectorized_dir，作为后续查看和排查的记录。
    config.vectorized_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "collection": config.milvus_collection_knowledge,
        "model": config.embedding_model_name,
        "dimension": embedder.dimension,
        "inserted_total": inserted_total,
        "files": manifest,
    }
    (config.vectorized_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# 命令行入口：可以用 --clean、--chunk、--embed 分别执行不同阶段。
def main() -> int:
    parser = argparse.ArgumentParser(description="Parse mental-health PDFs with all ingestion engines")
    parser.add_argument("--file", help="Process one PDF instead of the complete raw directory")
    parser.add_argument("--clean", action="store_true", help="Clean MinerU JSON results")
    parser.add_argument("--chunk", action="store_true", help="Chunk cleaned JSON results")
    parser.add_argument("--embed", action="store_true", help="Embed chunks and insert them into Milvus")
    args = parser.parse_args()

    # 读取 .env 里的配置，让模型路径、Milvus 地址等参数生效。
    load_dotenv()
    # 关闭一个 Paddle 相关的模型源检查，避免启动时额外检查网络模型源。
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # 根据环境变量创建入库配置，并校验配置是否合法。
    config = IngestionConfig.from_env()
    config.validate()
    configure_logging("INFO")
    pipeline = PdfIngestionPipeline(config)

    # --clean：把解析结果清洗成更适合后续处理的文本。
    if args.clean:
        cleaner = DocumentCleaner(config.chunk_size, config.chunk_overlap)
        cleaner.clean_directory(config.processed_dir, config.cleaned_dir)
        return 0
    # --chunk：把清洗后的文档切成多个 chunk。
    if args.chunk:
        chunker = DocumentChunker(config.chunk_size, config.chunk_overlap)
        chunker.chunk_directory(config.cleaned_dir, config.chunks_dir)
        return 0
    # --embed：生成向量并写入 Milvus，是进入向量库的关键命令。
    if args.embed:
        embed_chunks(config)
        return 0
    # 如果指定了单个 PDF，就只处理这一个文件。
    if args.file:
        from pathlib import Path

        pipeline.process_file(Path(args.file))
        return 0
    # 没有指定文件时，处理 raw 目录下的全部 PDF。
    pipeline.process_directory()
    return 0


# 直接运行这个文件时，从 main 开始执行命令行程序。
if __name__ == "__main__":
    raise SystemExit(main())
