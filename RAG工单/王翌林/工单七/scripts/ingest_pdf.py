# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
scripts/ingest_pdf.py — PDF 入库完整流水线

流程：PDF → 解析 → 分块 → 嵌入 → Milvus + MySQL

用法：
  python scripts/ingest_pdf.py --pdf "附件/招股说明书1.pdf"
  python scripts/ingest_pdf.py --pdf "附件/招股说明书1.pdf" --chunk-size 600 --top-k 10
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

# ========== 工单：人工智能NLP-RAG-基于PDF文档的问答系统 ==========
# 导入 src 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from loguru import logger

def main():
    parser = argparse.ArgumentParser(
        description="PDF 入库流水线（工单：人工智能NLP-RAG-基于PDF文档的问答系统）")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--chunk-size", type=int, default=600, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="分块重叠")
    parser.add_argument("--top-k", type=int, default=1000, help="Milvus 插入候选上限")
    parser.add_argument("--skip-parse", action="store_true", help="跳过解析（已有 parsed JSON）")
    parser.add_argument("--skip-chunk", action="store_true", help="跳过分块（已有 chunks JSON）")
    parser.add_argument("--skip-embed", action="store_true", help="跳过嵌入（已有向量）")
    parser.add_argument("--reset", action="store_true", help="重置 Milvus collection 后重新入库")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF 不存在: {pdf_path}"); sys.exit(1)

    stem = pdf_path.stem
    parsed_path = PROJECT_ROOT / "data" / "parsed" / f"{stem}.json"
    chunks_path = PROJECT_ROOT / "data" / "chunks" / f"{stem}_chunks.json"

    logger.info(f"{'='*60}")
    logger.info(f"📥 PDF 入库: {pdf_path.name}")
    logger.info(f"   chunk_size={args.chunk_size}, overlap={args.chunk_overlap}")
    logger.info(f"{'='*60}")

    total_t0 = time.time()

    # ========== Step 1: PDF 解析 ==========
    if not args.skip_parse:
        logger.info(f"[1/5] 正在解析 PDF → {parsed_path}")
        t0 = time.time()
        from src.pdf_parser import parse_pdf
        parsed = parse_pdf(str(pdf_path), extract_tables=True)
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        logger.info(f"  ✅ 解析完成: {parsed.get('total_pages',0)}页 / {len(parsed.get('text','')):,}字 | {time.time()-t0:.1f}s")
    else:
        logger.info(f"[1/5] 跳过解析（--skip-parse）")
        with open(parsed_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)

    # ========== Step 2: 分块 ==========
    if not args.skip_chunk:
        logger.info(f"[2/5] 正在分块 → {chunks_path}")
        t0 = time.time()
        from src.chunker import chunk_parsed_document
        chunked = chunk_parsed_document(parsed, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunked, f, ensure_ascii=False, indent=2)
        logger.info(f"  ✅ 分块完成: {chunked['total_chunks']}块 | {time.time()-t0:.1f}s")
    else:
        logger.info(f"[2/5] 跳过分块（--skip-chunk）")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunked = json.load(f)

    chunks_list = chunked["chunks"]

    # ========== Step 3: 嵌入 ==========
    if not args.skip_embed:
        logger.info(f"[3/5] 正在生成向量嵌入...")
        t0 = time.time()
        from src.embedding import get_embedder
        embedder = get_embedder()
        texts = [c.get("text", "") for c in chunks_list]
        vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)
        for i, c in enumerate(chunks_list):
            c["vector"] = vectors[i].tolist()
        logger.info(f"  ✅ 嵌入完成: {len(chunks_list)}×{embedder.dim} | {time.time()-t0:.1f}s")
    else:
        logger.info(f"[3/5] 跳过嵌入（--skip-embed）")

    # ========== Step 4: 入库 Milvus ==========
    logger.info(f"[4/5] 正在写入 Milvus 向量库...")
    t0 = time.time()
    from src.vector_store import VectorStore, import_chunks_json
    vs = VectorStore()
    if args.reset:
        logger.warning("  ⚠️ --reset: 重置 collection")
        vs.drop_collection()
    import_chunks_json(str(chunks_path), vs=vs)
    stats = vs.get_stats()
    logger.info(f"  ✅ Milvus: mode={stats.get('mode')}, entities={stats.get('num_entities')} | {time.time()-t0:.1f}s")

    # ========== Step 5: 入库 MySQL ==========
    logger.info(f"[5/5] 正在写入 MySQL 关系库...")
    t0 = time.time()
    try:
        from src.db import get_session
        from src.models import Document, Chunk
        file_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
        with get_session() as sess:
            doc = sess.query(Document).filter_by(file_hash=file_hash).first()
            if doc is None:
                doc = Document(
                    filename=pdf_path.name,
                    file_path=str(pdf_path),
                    file_hash=file_hash,
                    status="done",
                    total_pages=chunked.get("total_pages", 0),
                    total_chars=sum(c.get("char_count", 0) for c in chunks_list),
                    total_chunks=len(chunks_list),
                )
                sess.add(doc); sess.commit(); sess.refresh(doc)
            else:
                doc.total_chunks = len(chunks_list)
                doc.status = "done"
                sess.commit()
            # 清除旧 chunks（如果重新入库）
            old_chunks = sess.query(Chunk).filter_by(doc_id=doc.id).all()
            if old_chunks:
                logger.info(f"  清除旧 chunks: {len(old_chunks)} 条")
                for oc in old_chunks: sess.delete(oc)
                sess.commit()
            for ck in chunks_list:
                sess.add(Chunk(
                    doc_id=doc.id,
                    chunk_id=ck.get("chunk_id", ""),
                    chunk_index=ck.get("chunk_index", 0),
                    global_index=ck.get("global_index", 0),
                    content=ck.get("text", ""),
                    char_count=ck.get("char_count", 0),
                    page=ck.get("page", 0),
                ))
            sess.commit()
        logger.info(f"  ✅ MySQL: doc_id={doc.id}, chunks={len(chunks_list)} | {time.time()-t0:.1f}s")
    except Exception as e:
        logger.warning(f"  ⚠️ MySQL 入库跳过（不影响 Milvus）: {e}")

    total_s = time.time() - total_t0
    logger.info(f"\n🎉 入库完成！总耗时 {total_s:.1f}s")
    logger.info(f"   Milvus: {stats.get('num_entities', len(chunks_list))} vectors | mode={stats.get('mode')}")
    logger.info(f"   MySQL:  doc_id={doc.id} | {len(chunks_list)} chunks")

if __name__ == "__main__":
    main()
