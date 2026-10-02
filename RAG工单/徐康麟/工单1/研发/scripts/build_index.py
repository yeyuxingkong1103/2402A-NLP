"""建立/重建知识库索引。

用法::

    # 全量建索引（默认使用 data/raw/招股说明书1.pdf）
    python scripts/build_index.py

    # 只解析前 120 页（快速验证链路）
    python scripts/build_index.py --max-pages 120

    # 指定 PDF 与文档 ID
    python scripts/build_index.py --pdf data/raw/招股说明书1.pdf --doc-id doc_demo

    # 只做解析与分块、不做向量化（用于快速检查分块质量）
    python scripts/build_index.py --parse-only

输出：
- data/processed/*.parsed.json / *.pages.txt / *.tables.jsonl / chunks.jsonl
- data/index/vectors.npy + vectors_meta.jsonl（或 data/index/chroma/）
- data/index/bm25_index.pkl
- data/index/rag.sqlite3（documents / chunks 表）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许以 `python 研发/scripts/build_index.py` 方式直接运行：
# 本文件位于 <root>/研发/scripts/，需要把「研发」加入 sys.path 才能 import app.*
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.core.chunker import Chunker  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging_conf import log_stage, logger, setup_logging  # noqa: E402
from app.core.pdf_parser import PDFParser, parser_capabilities  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="建立 RAG 知识库索引")
    parser.add_argument("--pdf", type=str, default=str(settings.paths.default_pdf), help="PDF 文件路径")
    parser.add_argument("--doc-id", type=str, default=None, help="文档 ID（默认按文件内容生成）")
    parser.add_argument("--max-pages", type=int, default=0, help="最多解析页数，0 表示全部")
    parser.add_argument("--parse-only", action="store_true", help="只解析与分块，不建向量索引")
    parser.add_argument("--no-tables", action="store_true", help="跳过表格提取（更快）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果摘要")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    started = time.perf_counter()

    log_stage("阶段1", "开始建索引", pdf=args.pdf, max_pages=args.max_pages or "全部")
    logger.info("scripts.build_index", "解析器能力", **parser_capabilities())

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("scripts.build_index", "PDF 不存在", path=str(pdf_path))
        print(f"[错误] PDF 不存在: {pdf_path}", file=sys.stderr)
        return 2

    # ---- 解析 ----
    pdf_parser = PDFParser(settings)
    document = pdf_parser.parse(
        pdf_path, doc_id=args.doc_id, max_pages=args.max_pages, extract_tables=not args.no_tables
    )
    saved = pdf_parser.save(document)
    log_stage(
        "阶段1",
        "PDF 解析完成",
        pages=document.page_count,
        tables=len(document.tables),
        chars=sum(page.char_count for page in document.pages),
    )

    # ---- 分块 ----
    chunker = Chunker(settings)
    chunks = chunker.split(document)
    chunk_path = chunker.save(chunks)
    stats = {
        "text_chunks": sum(1 for chunk in chunks if chunk.type == "text"),
        "table_chunks": sum(1 for chunk in chunks if chunk.type == "table"),
        "avg_chars": round(sum(chunk.char_count for chunk in chunks) / len(chunks), 1) if chunks else 0,
    }
    log_stage("阶段1", "分块完成", total=len(chunks), **stats)

    result: dict[str, object] = {
        "doc_id": document.doc_id,
        "title": document.title,
        "pages": document.page_count,
        "tables": len(document.tables),
        "chunks": len(chunks),
        **stats,
        "processed": {key: str(value) for key, value in saved.items()},
        "chunks_path": str(chunk_path),
    }

    # ---- 向量化 + BM25 + SQLite ----
    if not args.parse_only:
        from app.core.qa_engine import QAEngine  # 延迟导入，避免 --parse-only 时加载重依赖

        engine = QAEngine(doc_id=None, auto_load_index=False)
        index_info = engine.retriever.build_index(chunks)
        index_paths = engine.retriever.save_index()

        from app.models.schemas import DocumentMeta

        engine.store.upsert_document(
            DocumentMeta(
                doc_id=document.doc_id,
                title=document.title,
                source_path=document.source_path,
                page_count=document.page_count,
                chunk_count=len(chunks),
                table_count=len(document.tables),
                status="indexed",
                is_default=True,
            )
        )
        engine.store.insert_chunks(chunks, replace_doc=True)
        result.update(
            {
                "vector_count": index_info.get("vector", 0),
                "dimension": index_info.get("dimension", 0),
                "embedder": index_info.get("embedder", ""),
                "vector_backend": engine.vector_store.name,
                "index_paths": index_paths,
                "sqlite": str(settings.paths.sqlite_path),
            }
        )
        log_stage(
            "阶段1",
            "索引构建完成",
            vectors=index_info.get("vector", 0),
            dimension=index_info.get("dimension", 0),
            embedder=index_info.get("embedder", ""),
            bm25=index_info.get("bm25", 0),
        )

    result["elapsed_s"] = round(time.perf_counter() - started, 2)
    log_stage("阶段1", "建索引结束", elapsed_s=result["elapsed_s"])

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n===== 建索引结果 =====")
        for key, value in result.items():
            print(f"{key:>16}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
