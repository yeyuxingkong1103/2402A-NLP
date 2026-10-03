"""索引构建脚本：PDF → 解析 → 分块 → 嵌入 → 向量索引 + BM25 + SQLite。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 脚本（对应 设计/接口设计.md §2.18）

用法（工作目录 = E:\\gao6gongdan\\工单2）::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --max-pages 40   # 小样本冒烟
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --rebuild       # 强制重建

退出码：0 成功；2 业务失败（PDF 解析/索引构建失败）；3 参数错误。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 允许脚本直接从仓库源码目录导入 app 包
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.core.config import get_settings, rejected_env_keys  # noqa: E402
from app.core.errors import PDFParseError, RAGError  # noqa: E402
from app.core.logging_conf import flush_logs, log_stage, logger, setup_logging  # noqa: E402
from app.core.pdf_parser import parser_capabilities  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="构建 RAG 索引（工单2）")
    parser.add_argument("--pdf", default="", help="PDF 路径，默认 研发/data/raw/招股说明书1.pdf")
    parser.add_argument("--embed-model", default="", help="覆盖 Ollama 嵌入模型名（默认 bge-m3:latest）")
    parser.add_argument("--rebuild", action="store_true", help="强制重建（清空现有索引）")
    parser.add_argument("--max-pages", type=int, default=0, help="仅解析前 N 页（冒烟测试用，0=全部）")
    parser.add_argument("--reuse-processed", action="store_true", help="复用已有 chunks.jsonl，跳过 PDF 解析")
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本入口。"""
    args = build_parser().parse_args(argv)
    if args.max_pages < 0:
        print("参数错误：--max-pages 不能为负数", file=sys.stderr)
        return 3
    if args.embed_model:
        os.environ["RAG_EMBEDDING__OLLAMA_MODEL"] = args.embed_model
        os.environ["RAG_EMBEDDING__BACKEND"] = "ollama"

    setup_logging()
    settings = get_settings()
    from app.core.chunker import Chunker
    from app.core.embedder import get_embedder
    from app.core.qa_engine import QAEngine
    from app.core.retriever import Retriever
    from app.core.reranker import get_reranker
    from app.core.vector_store import VectorStore

    pdf_path = Path(args.pdf) if args.pdf else settings.paths.default_pdf
    print("=" * 78)
    print("工单2 索引构建：人工智能NLP-RAG-基于PDF文档的问答系统优化")
    print(f"PDF          : {pdf_path}")
    print(f"嵌入模型     : {settings.embedding.ollama_model}（期望 {settings.embedding.dimension} 维）")
    print(f"解析能力     : {parser_capabilities()}")
    print(f"分块参数     : chunk_size={settings.chunk.chunk_size} overlap={settings.chunk.chunk_overlap}")
    print(f"LLM 后端配置 : {settings.llm.backend} / model={settings.llm.model}")
    print(f"索引根目录   : {settings.paths.data_index}")
    print(f"SQLite       : {settings.paths.sqlite_path}")
    if rejected_env_keys():
        print(f"被忽略的环境变量: {rejected_env_keys()}")
    print("=" * 78)

    started = time.perf_counter()
    try:
        log_stage("index", "开始构建索引", pdf=str(pdf_path), rebuild=args.rebuild)

        if args.reuse_processed:
            # 复用已有 chunks.jsonl（只重建向量/BM25/SQLite）
            from app.core.chunker import Chunker as _Chunker

            chunks_path = settings.paths.data_processed / "chunks.jsonl"
            if not chunks_path.exists():
                print(f"业务失败：找不到 {chunks_path}，无法复用", file=sys.stderr)
                return 2
            chunks = _Chunker.load(chunks_path)
            embedder = get_embedder()
            store = VectorStore(index_dir=settings.index_dir(embedder.slug, embedder.dimension))
            retriever = Retriever(vector_store=store, reranker=get_reranker())
            print(f"[1/2] 已复用分块 {len(chunks)} 块（跳过 PDF 解析）")
            stats = retriever.build_index(chunks, reset=True)
            retriever.save_index()
            engine = QAEngine()
            engine._store.upsert_document(_document_meta(settings, pdf_path, chunks))
            engine._store.insert_chunks(chunks, replace_doc=True)
        else:
            if not pdf_path.exists():
                print(f"业务失败：PDF 不存在 -> {pdf_path}", file=sys.stderr)
                return 2
            engine = QAEngine()
            stats = engine.build_index(pdf_path=pdf_path, reset=True)

        elapsed = (time.perf_counter() - started) / 60
        print("-" * 78)
        print("构建完成，关键规模：")
        for key in (
            "doc_id",
            "pages",
            "tables",
            "merged_tables",
            "table_errors",
            "blocks",
            "chunks",
            "vectors",
            "dimension",
            "bm25_docs",
            "chunks_written",
        ):
            if key in stats:
                print(f"  {key:<16}: {stats[key]}")
        if "chunk_stats" in stats:
            print(f"  chunk_stats     : {json.dumps(stats['chunk_stats'], ensure_ascii=False)}")
        if "embed_ms" in stats:
            print(f"  嵌入耗时        : {stats['embed_ms']} ms")
        print(f"  索引目录        : {stats.get('index_dir')}")
        print(f"  总耗时          : {elapsed:.2f} 分钟")
        print("-" * 78)
        print("复现命令：pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py")
        log_stage("index", "索引构建结束", elapsed_min=round(elapsed, 2), stats=stats)
        return 0
    except (PDFParseError, RAGError) as exc:
        logger.exception("scripts.build_index", "索引构建业务失败", code=exc.code)
        print(f"业务失败[{exc.code}]：{exc.message}", file=sys.stderr)
        return 2
    except Exception as exc:  # 兜底：不允许静默失败
        logger.exception("scripts.build_index", "索引构建未预期异常")
        print(f"构建异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        flush_logs()


def _document_meta(settings, pdf_path: Path, chunks: list):
    """构造文档元数据（复用分块路径用）。"""
    from app.models.schemas import DocumentMeta

    return DocumentMeta(
        doc_id=pdf_path.stem,
        title=pdf_path.stem,
        source_path=str(pdf_path),
        page_count=max((chunk.page for chunk in chunks), default=0),
        chunk_count=len(chunks),
        table_count=sum(1 for chunk in chunks if chunk.type == "table"),
        status="indexed",
        is_default=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
