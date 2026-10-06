# -*- coding: utf-8 -*-
"""T4 建索引驱动：解析 → 分块 → 嵌入（Ollama/本地降级）→ 向量索引 + 自实现 BM25 → manifest。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
契约：设计/接口设计.md §6.1（CLI）、§5.4（index_manifest.json）、§7 索引目录布局

用法（工作目录 = E:\\gao6gongdan\\工单3）：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --no-embed      # 仅 BM25（调试）
    pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --reuse-chunks  # 复用已有分块

索引目录：``{index-dir}/{model_slug}/``（如 ``研发/data/index/bge-m3_1024/``），内含
``vectors.npy``、``ids.json``、``bm25.pkl``、``bm25.meta.json``、``index_manifest.json``。

退出码：0 成功；1 校验失败（无 PDF / 页数不符 / 块数不一致）；2 环境或入参错误。
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

import numpy as np  # noqa: E402

from app.core import chunker, embedder, pdf_parser, vector_store  # noqa: E402
from app.core.bm25_index import BM25Index  # noqa: E402
from app.core.config import describe_runtime, discover_pdfs, get_config, model_slug  # noqa: E402
from app.core.errors import RagError  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.text_utils import read_jsonl  # noqa: E402

# 两份语料的期望页数（用于「页数与预期不符」校验；按发现顺序核对，不硬编码文件名）
EXPECTED_PAGES = {548, 350}
MANIFEST_FILE = "index_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数（与设计 §6.1 一致，另加 --reuse-chunks 复用分块以省去重复解析）。"""
    parser = argparse.ArgumentParser(
        description="工单3 建索引（人工智能NLP-RAG-PDF文档的表格解析及检索优化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-dir", default=None, help="语料目录，默认 研发/data/raw")
    parser.add_argument("--out-dir", default=None, help="JSONL 产物目录，默认 研发/data/processed")
    parser.add_argument("--index-dir", default=None, help="索引根，默认 研发/data/index")
    parser.add_argument("--no-embed", action="store_true", help="只建 BM25（manifest.embedding.backend=none）")
    parser.add_argument("--no-table", action="store_true", help="关表格扫描（仅调试，会打 warnings）")
    parser.add_argument("--fast-table", action="store_true", help="跳过「无绘制线且文本极短」页的表扫描")
    parser.add_argument("--force", action="store_true", help="重建（覆盖产物与索引）")
    parser.add_argument("--limit-pages", type=int, default=None, help="只跑前 N 页（调试用，会打 warnings）")
    parser.add_argument("--reuse-chunks", action="store_true", help="复用 out-dir/chunks.jsonl，跳过解析")
    parser.add_argument("--batch-size", type=int, default=None, help="嵌入批大小（默认 RAG_EMBEDDING__BATCH_SIZE=16）")
    parser.add_argument("--log-level", default=None, help="RAG_LOG__LEVEL 覆盖")
    return parser


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_existing_chunks(out_dir: Path) -> list[chunker.Chunk]:
    """从 ``chunks.jsonl`` 复用已有分块（按 order 排序）。"""
    path = out_dir / "chunks.jsonl"
    if not path.is_file():
        raise RagError(f"复用分块失败：找不到 {path}", code="RAG-2200", stage="chunk")
    chunks = [chunker.Chunk.from_dict(row) for row in read_jsonl(path)]
    chunks.sort(key=lambda c: c.order)
    return chunks


def _fast_table_pages(source: Any, limit_pages: int | None, log: Any) -> set[int]:
    """``--fast-table``：跳过「无绘制线且文本极短」的页（文本 < 40 字符且 get_drawings 为空）。"""
    skip: set[int] = set()
    import pymupdf

    doc = pymupdf.open(str(source.path))
    try:
        total = int(doc.page_count)
        wanted = range(1, (limit_pages or total) + 1)
        for page_no in wanted:
            page = doc.load_page(page_no - 1)
            text = "".join((page.get_text("text") or "").split())
            if len(text) < 40:
                try:
                    has_drawings = len(page.get_drawings()) > 0
                except Exception:  # noqa: BLE001 —— 取不到绘制线时保守处理：不跳过（留痕）
                    has_drawings = True
                    log.log_event("fast_table.drawings_error", level="WARNING", page=page_no,
                                  degrade="保留该页表扫描")
                if not has_drawings:
                    skip.add(page_no)
    finally:
        doc.close()
    log.log_event("fast_table.scan", file_name=source.file_name, skipped=len(skip),
                  pages=sorted(skip)[:30])
    # 返回「仍需扫描表格」的页清单（1-based）
    return {p for p in range(1, int(total) + 1) if p not in skip}


def main(argv: list[str] | None = None) -> int:
    """执行建索引全流程并打印实测摘要。"""
    args = build_parser().parse_args(argv)
    cfg = get_config()
    if args.log_level:
        cfg = get_config(refresh=True)
    setup_logging(cfg, run_id=cfg.run_id, force=True)
    atexit.register(shutdown_logging)
    log = get_logger("build_index")
    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.paths.raw_dir
    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.processed_dir
    index_root = (Path(args.index_dir) if args.index_dir else cfg.paths.index_dir).resolve()
    batch_size = int(args.batch_size or cfg.embedding.batch_size)

    with log.enter("build_index", {"raw_dir": str(raw_dir), "out_dir": str(out_dir),
                                   "index_dir": str(index_root), "no_embed": args.no_embed,
                                   "no_table": args.no_table, "fast_table": args.fast_table,
                                   "reuse_chunks": args.reuse_chunks,
                                   "limit_pages": args.limit_pages, "batch_size": batch_size}) as span:
        print("=" * 78)
        print(f"T4 建索引｜{describe_runtime()['work_order']}")
        print(f"解释器：{sys.executable}（{sys.version.split()[0]}）")
        print(f"语料  ：{raw_dir}　产物：{out_dir}　索引根：{index_root}")
        print("=" * 78)
        t_all = time.perf_counter()
        warnings: list[str] = []
        if args.no_table:
            warnings.append("no_table：关闭了表格扫描（仅调试）")
        if args.limit_pages:
            warnings.append(f"limit_pages={args.limit_pages}（仅调试，不得用于验收）")
        if args.no_embed:
            warnings.append("no_embed：仅构建 BM25 索引（embedding.backend=none）")

        if not raw_dir.is_dir():
            print(f"❌ 语料目录不存在：{raw_dir}")
            return 2
        sources = discover_pdfs(raw_dir, logger=log)
        if not sources:
            print(f"❌ 目录下没有任何 PDF：{raw_dir}")
            return 1
        print(f"发现 {len(sources)} 份 PDF：" + "、".join(f"{s.file_name}({s.page_count}页)" for s in sources))

        # ---- 1) 解析 + 分块（或复用） ----
        parse_results: list[Any] = []
        chunks: list[chunker.Chunk] = []
        chunks_path = out_dir / "chunks.jsonl"
        if args.reuse_chunks and chunks_path.is_file() and not args.force:
            chunks = _load_existing_chunks(out_dir)
            print(f"复用已有分块：{chunks_path} → {len(chunks)} 块（跳过解析）")
            log.log_event("build_index.reuse_chunks", path=str(chunks_path), chunks=len(chunks))
        else:
            if chunks_path.exists():
                chunks_path.unlink()
            pages_arg = list(range(1, args.limit_pages + 1)) if args.limit_pages else None
            for source in sources:
                if not source.readable:
                    log.log_event("build_index.skip_unreadable", level="ERROR", file_name=source.file_name)
                    print(f"  ⚠️ {source.file_name} 不可读，已跳过")
                    continue
                table_pages = None
                if args.fast_table and not args.no_table:
                    table_pages = _fast_table_pages(source, args.limit_pages, log)
                result = pdf_parser.parse_pdf(
                    source, cfg=cfg, scan_tables=not args.no_table,
                    pages=pages_arg, table_pages=table_pages, logger=log,
                )
                pdf_parser.write_parse_artifacts(result, out_dir, overwrite=True)
                pdf_parser.persist_parse_result(result, db_path=index_root / "rag.sqlite3",
                                                source=source, logger=log)
                file_chunks = chunker.build_chunks(result, cfg=cfg, logger=log)
                chunker.persist_chunks(file_chunks, db_path=index_root / "rag.sqlite3", logger=log)
                chunks.extend(file_chunks)
                parse_results.append(result)
                print(f"  ✅ {result.file_name}：{result.page_count} 页 → {len(file_chunks)} 块"
                      f"（表块 {sum(1 for c in file_chunks if c.type == 'table')}）"
                      f"，{result.elapsed_ms / 1000:.1f}s")
            chunker.write_chunks(chunks, chunks_path, overwrite=True)
        if not chunks:
            print("❌ 没有任何分块，无法建索引")
            return 1
        text_chunks = sum(1 for c in chunks if c.type == "text")
        table_chunks = sum(1 for c in chunks if c.type == "table")
        print(f"分块合计：{len(chunks)}（文本 {text_chunks} / 表格 {table_chunks}）")

        # ---- 2) 嵌入（可跳过） ----
        vectors: np.ndarray | None = None
        embedding_info: dict[str, Any] = {"backend": "none", "model": cfg.llm.ollama_embed_model,
                                          "dim": 0, "count": 0, "elapsed_ms": 0.0}
        if args.no_embed:
            print("⚠️ --no-embed：跳过嵌入，仅建 BM25")
            slug = "bm25_only"
        else:
            print(f"开始嵌入：{len(chunks)} 块，batch_size={batch_size}，模型={cfg.llm.ollama_embed_model}")
            embed_started = time.perf_counter()
            result = embedder.embed_texts([c.content for c in chunks], cfg=cfg,
                                          batch_size=batch_size, logger=log)
            vectors = result.vectors
            embedding_info = {"backend": result.backend, "model": result.model, "dim": result.dim,
                              "count": result.count, "elapsed_ms": result.elapsed_ms}
            if result.count != len(chunks):
                print(f"❌ 向量条数 {result.count} 与块数 {len(chunks)} 不一致")
                return 1
            slug = model_slug(result.model, result.dim)
            print(f"嵌入完成：{result.count} 条 × {result.dim} 维，{result.elapsed_ms / 1000:.1f}s"
                  f"（backend={result.backend}，墙钟 {time.perf_counter() - embed_started:.1f}s）")

        model_dir = index_root / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        # ---- 3) 向量索引 ----
        vector_info: dict[str, Any] = {"backend": "none", "count": 0, "dim": 0}
        if vectors is not None:
            vector_info = vector_store.build_vector_index(
                chunks, vectors, index_dir=model_dir, model=cfg.llm.ollama_embed_model,
                logger=log, use_faiss=cfg.vector_store.use_faiss,
            )
            print(f"向量索引：{model_dir}（{vector_info['count']} × {vector_info['dim']}"
                  f"，faiss={'是' if vector_info.get('faiss') else '否'}）")

        # ---- 4) BM25（自实现） ----
        bm25 = BM25Index(k1=1.5, b=0.75, use_jieba=True)
        bm25.build(chunks, logger=log)
        dim_model = (f"{embedding_info.get('model')}#{embedding_info.get('dim')}"
                     if vectors is not None else None)
        bm25.save(model_dir, dim_model=dim_model, logger=log)
        print(f"BM25 索引：{bm25.size()} 块，词表 {bm25.vocab_size()} 词 → {model_dir / 'bm25.pkl'}")

        # ---- 5) manifest（设计 §5.4） ----
        manifest = _build_manifest(
            cfg=cfg, sources=sources, parse_results=parse_results, chunks=chunks,
            embedding_info=embedding_info, bm25=bm25, model_dir=model_dir,
            warnings=warnings, elapsed_ms=round((time.perf_counter() - t_all) * 1000, 2),
            reused=bool(args.reuse_chunks and not parse_results),
        )
        manifest_path = model_dir / MANIFEST_FILE
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"manifest：{manifest_path}")

        # ---- 6) 校验 ----
        if not args.limit_pages and parse_results:
            found = {r.page_count for r in parse_results}
            if not found <= EXPECTED_PAGES:
                print(f"❌ 页数与预期不符：实际 {sorted(found)}，预期 ⊆ {sorted(EXPECTED_PAGES)}")
                return 1
            print(f"✅ 页数校验通过：{sorted(found)} ⊆ {sorted(EXPECTED_PAGES)}")
        print("-" * 78)
        print(f"总耗时：{time.perf_counter() - t_all:.1f}s｜块数 {len(chunks)}｜"
              f"向量 {vector_info.get('count', 0)}×{vector_info.get('dim', 0)}｜BM25 词表 {bm25.vocab_size()}")
        if warnings:
            print("⚠️ warnings：" + "；".join(warnings))
        span.set_output({"files": len(sources), "chunks": len(chunks), "text_chunks": text_chunks,
                         "table_chunks": table_chunks, "embedding": embedding_info,
                         "bm25": {"count": bm25.size(), "vocab_size": bm25.vocab_size()},
                         "index_dir": str(model_dir), "warnings": warnings})
        return 0


def _build_manifest(*, cfg: Any, sources: Sequence[Any], parse_results: Sequence[Any],
                    chunks: Sequence[chunker.Chunk], embedding_info: dict[str, Any],
                    bm25: BM25Index, model_dir: Path, warnings: Sequence[str],
                    elapsed_ms: float, reused: bool) -> dict[str, Any]:
    """按设计 §5.4 组装 index_manifest.json（数字全部来自实测，不照抄示例）。"""
    per_file: list[dict[str, Any]] = []
    for source in sources:
        result = next((r for r in parse_results if r.file_name == source.file_name), None)
        file_chunks = [c for c in chunks if c.file_name == source.file_name]
        entry: dict[str, Any] = {
            "file_name": source.file_name,
            "sha256_16": source.sha256_16,
            "page_count": (result.page_count if result is not None else source.page_count),
            "text_chunks": sum(1 for c in file_chunks if c.type == "text"),
            "table_chunks": sum(1 for c in file_chunks if c.type == "table"),
        }
        if result is not None:
            stats = result.stats
            entry.update({
                "tables": len(result.tables), "raw_tables": stats.get("raw_tables"),
                "continued_tables": stats.get("continued_tables"),
                "repeated_headers_removed": stats.get("repeated_headers_removed"),
                "degenerate_tables": stats.get("degenerate_tables"),
                "text_blocks": len(result.text_blocks), "max_table_run": stats.get("max_run"),
                "table_pages": stats.get("table_pages"),
            })
        else:
            entry["note"] = "reuse_chunks：未重新解析，逐文件解析统计见 data/processed/*.jsonl"
        per_file.append(entry)

    deps, dep_warnings = _deps_report()
    return {
        "run_id": cfg.run_id,
        "created_at": _now_iso(),
        "files": per_file,
        "embedding": {
            "backend": embedding_info.get("backend", "none"),
            "model": embedding_info.get("model", ""),
            "dim": embedding_info.get("dim", 0),
            "count": embedding_info.get("count", 0),
            "elapsed_ms": embedding_info.get("elapsed_ms", 0.0),
        },
        "bm25": {"count": bm25.size(), "vocab_size": bm25.vocab_size(),
                 "k1": bm25.k1, "b": bm25.b,
                 "table_chunks": sum(1 for m in bm25.meta.values() if m.get("type") == "table")},
        "index_dir": _repo_rel(model_dir),
        "elapsed_ms": elapsed_ms,
        "total_chunks": len(chunks),
        "reused_chunks": reused,
        "deps": deps,
        "warnings": list(warnings) + dep_warnings,
    }


def _dep_version(name: str) -> str:
    """依赖版本（**不可用即返回 "unavailable"**，绝不把「装有但 import 失败」写成可用）。

    实测教训（环境事实 §8.4）：``loguru`` 的 ``find_spec`` 命中、元数据版本 0.7.3，
    但 ``import loguru`` 抛 ``ModuleNotFoundError: No module named 'win32_setctime'``。
    只读元数据的写法会让读者误判为可用 —— 因此元数据存在后必须**再真实 import 一次**，
    失败则返回 ``<版本>(present_but_unimportable:<异常>)``（并在 manifest.warnings 写明原因）。
    """
    aliases = {"pymupdf": ("pymupdf", "PyMuPDF"), "faiss": ("faiss-cpu", "faiss")}
    version = "unavailable"
    try:
        import importlib.metadata as md

        for candidate in aliases.get(name, (name,)):
            try:
                version = md.version(candidate)
                break
            except md.PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001 —— 版本探测失败不能阻断建索引，但如实标注
        version = "unavailable"
    if version == "unavailable":
        return version
    try:
        import importlib

        importlib.import_module(name)
        return version
    except Exception as exc:  # noqa: BLE001 —— 半残包必须显式标注，不静默
        return f"{version}(present_but_unimportable:{type(exc).__name__})"


def _deps_report() -> tuple[dict[str, str], list[str]]:
    """依赖清单 + 由其诚实标注衍生的 warnings（供 manifest 使用）。"""
    names = ("pymupdf", "numpy", "jieba", "faiss", "loguru", "rank_bm25")
    deps = {name: _dep_version(name) for name in names}
    warnings: list[str] = []
    for name, value in deps.items():
        if "present_but_unimportable" in value:
            warnings.append(f"依赖 {name} 元数据存在但真实 import 失败：{value}（按不可用对待）")
        elif value == "unavailable" and name in {"loguru", "rank_bm25"}:
            warnings.append(f"依赖 {name} 不可用（本仓库已按要求自实现替代）")
    return deps, warnings


def _repo_rel(path: Path) -> str:
    """相对仓库根的 POSIX 路径（不在仓库内则返回绝对路径，不抛异常）。"""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(Path(path).resolve()).replace("\\", "/")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RagError as exc:
        print(f"\n❌ 建索引失败：{exc}", file=sys.stderr)
        print(f"   detail={exc.detail}", file=sys.stderr)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底：完整堆栈 + 退出码 2
        import traceback

        traceback.print_exc()
        raise SystemExit(2)
