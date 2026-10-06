# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
scripts/ingest_text_v3.py —— 工单三多文档文本入库

将 data/parsed_v3/{doc_name}_text.json 中的 text_chunks 入库到 Milvus rag_chunks collection。
支持多文档：招股说明书1 + 招股说明书2。

用法：
  python scripts/ingest_text_v3.py --rebuild
  python scripts/ingest_text_v3.py  # 增量（不删旧数据）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 工单三：确保项目根目录在 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# 工单三：显存优化环境变量
os.environ.setdefault("RAG_EMBED_DEVICE", "cpu")
os.environ.setdefault("RAG_EMBED_BATCH_SIZE", "8")
os.environ.setdefault("RAG_EMBED_MAX_SEQ", "512")

from src.embedding import get_embedder, DEFAULT_BATCH_SIZE
from src.vector_store import VectorStore


def load_text_chunks(parsed_dir: Path) -> list:
    """工单三：加载所有 parsed_v3 文本 chunk"""
    all_chunks = []
    for f in sorted(parsed_dir.glob("*_text.json")):
        if f.name == "ingest_summary.json":
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        doc_name = data.get("doc_name", f.stem)
        # 工单三：统一用 doc_name 作为 doc_id（与 rag_tables 一致）
        doc_id = doc_name
        company = data.get("company", "")
        chunks = data.get("text_chunks", [])
        for i, ch in enumerate(chunks):
            ch["doc_id"] = doc_id  # 覆盖 hash，统一用 doc_name
            ch.setdefault("doc_name", doc_name)
            ch.setdefault("company", company)
            ch.setdefault("chunk_index", i)
            ch.setdefault("source", doc_name)
            # 工单三：确保 text 字段存在（vector_store.insert_chunks 用 ch["text"]）
            if "text" not in ch and "content" in ch:
                ch["text"] = ch["content"]
        logger.info(f"  {doc_name}: {len(chunks)} text_chunks")
        all_chunks.extend(chunks)
    return all_chunks


def ingest_text_chunks(chunks: list, rebuild: bool = False) -> int:
    """工单三：文本 chunk 向量化 + 入库 rag_chunks"""
    if not chunks:
        logger.warning("无 text_chunks 可入库")
        return 0

    vs = VectorStore()
    if rebuild:
        logger.info(f"[工单三] 重建 rag_chunks collection")
        vs.drop_collection()

    vs.ensure_collection()

    # 工单三：批量向量化
    embedder = get_embedder()
    texts = [ch.get("text", "") for ch in chunks]
    logger.info(f"[工单三] 开始向量化: {len(texts)} chunks, "
                f"batch_size={DEFAULT_BATCH_SIZE}")
    t0 = time.time()
    vectors = embedder.encode(texts, batch_size=DEFAULT_BATCH_SIZE,
                              show_progress_bar=True)
    embed_ms = (time.time() - t0) * 1000
    logger.info(f"[工单三] 向量化完成: {embed_ms:.0f}ms")

    # 入库
    inserted = vs.insert_chunks(chunks, vectors)
    stats = vs.get_stats()
    logger.info(f"[工单三] 入库完成: {inserted} 条 → rag_chunks, "
                f"total={stats.get('num_entities')}")
    vs.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(
        description="工单三多文档文本入库（人工智能NLP-RAG-PDF文档的表格解析及检索优化）"
    )
    parser.add_argument("--parsed-dir", default="data/parsed_v3",
                        help="parsed_v3 目录")
    parser.add_argument("--rebuild", action="store_true",
                        help="重建 collection（先删后建）")
    args = parser.parse_args()

    parsed_dir = PROJECT_ROOT / args.parsed_dir
    if not parsed_dir.exists():
        logger.error(f"目录不存在: {parsed_dir}")
        sys.exit(1)

    logger.info(f"[工单三] 加载 text_chunks from {parsed_dir}")
    chunks = load_text_chunks(parsed_dir)
    logger.info(f"[工单三] 共 {len(chunks)} text_chunks 待入库")

    total = ingest_text_chunks(chunks, rebuild=args.rebuild)
    logger.info(f"[工单三] 完成：{total} 条文本 chunk 入库 rag_chunks")


if __name__ == "__main__":
    main()
