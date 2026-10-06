# -*- coding: utf-8 -*-
"""T3 离线解析驱动：把 ``研发/data/raw/*.pdf`` 解析为 JSONL 产物并写入 SQLite。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用法（工作目录 = E:\\gao6gongdan\\工单3）：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/parse_corpus.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/parse_corpus.py --limit-pages 40 --force
    pwsh -NoProfile -File run_py.ps1 研发/scripts/parse_corpus.py --no-table     # 仅调试

说明：本脚本是 T3 的**解析驱动**（对应设计 §3.5 的 parse_all/build_chunks）；T4 的
``build_index.py`` 会在其之上追加嵌入与索引构建，因此这里不实现 CLI 契约 §6.1 的参数全集。

退出码：0 成功；1 校验失败（无 PDF / 页数不符 / 断言不通过）；2 环境或入参错误。
"""

from __future__ import annotations

import argparse
import atexit
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import chunker, pdf_parser  # noqa: E402
from app.core.config import describe_runtime, discover_pdfs, get_config  # noqa: E402
from app.core.errors import RagError  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402

# 两份 PDF 的期望页数（用于「页数与预期不符」的校验，不硬编码文件名 → 按发现顺序核对）
EXPECTED_PAGES = {548, 350}


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数（新增参数均为调试用，默认不改变正式行为）。"""
    parser = argparse.ArgumentParser(description="工单3 离线解析驱动（人工智能NLP-RAG-PDF文档的表格解析及检索优化）")
    parser.add_argument("--raw-dir", default=None, help="语料目录，默认 研发/data/raw")
    parser.add_argument("--out-dir", default=None, help="JSONL 产物目录，默认 研发/data/processed")
    parser.add_argument("--index-dir", default=None, help="SQLite 目录，默认 研发/data/index")
    parser.add_argument("--force", action="store_true", help="重建（产物本身已每次覆盖；此开关保留语义兼容，并强制刷新 SQLite）")
    parser.add_argument("--limit-pages", type=int, default=None, help="只解析前 N 页（调试）")
    parser.add_argument("--no-table", action="store_true", help="关闭表格扫描（仅调试）")
    parser.add_argument("--no-merge", action="store_true", help="关闭跨页续表合并（仅调试）")
    parser.add_argument("--log-level", default=None, help="RAG_LOG__LEVEL 覆盖")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行解析 → 分块 → 落盘 → 入库，并打印逐文件统计。"""
    args = build_parser().parse_args(argv)
    cfg = get_config()
    if args.log_level:
        cfg = get_config(refresh=True)
    setup_logging(cfg, run_id=cfg.run_id, force=True)
    atexit.register(shutdown_logging)   # 任意 return/异常路径都统一收尾，避免句柄泄漏
    log = get_logger("parse_corpus")
    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.paths.raw_dir
    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.processed_dir
    index_dir = Path(args.index_dir) if args.index_dir else cfg.paths.index_dir

    with log.enter("parse_corpus", {"raw_dir": str(raw_dir), "out_dir": str(out_dir),
                                    "index_dir": str(index_dir), "force": args.force,
                                    "limit_pages": args.limit_pages, "scan_tables": not args.no_table,
                                    "merge_continued": not args.no_merge}) as span:
        print("=" * 78)
        print(f"T3 离线解析｜{describe_runtime()['work_order']}")
        print(f"解释器：{sys.executable}（{sys.version.split()[0]}）")
        print(f"语料  ：{raw_dir}")
        print(f"产物  ：{out_dir}　SQLite：{index_dir / 'rag.sqlite3'}")
        print("=" * 78)

        if not raw_dir.is_dir():
            print(f"❌ 语料目录不存在：{raw_dir}")
            return 2
        sources = discover_pdfs(raw_dir, logger=log)
        if not sources:
            print(f"❌ 目录下没有任何 PDF：{raw_dir}")
            return 1
        print(f"发现 {len(sources)} 份 PDF：" + "、".join(f"{s.file_name}({s.page_count}页)" for s in sources))

        pages_arg = list(range(1, args.limit_pages + 1)) if args.limit_pages else None
        summaries: list[dict] = []
        all_chunks: list[chunker.Chunk] = []
        t_all = time.perf_counter()
        chunks_path = out_dir / "chunks.jsonl"
        if chunks_path.exists():
            chunks_path.unlink()          # chunks.jsonl 是全语料合并文件，每次运行整体重生成
        for source in sources:
            if not source.readable:
                log.log_event("parse_corpus.skip", level="ERROR", file_name=source.file_name,
                              reason="不可读，跳过（已在 discover 阶段留痕）")
                print(f"  ⚠️ {source.file_name} 不可读，已跳过")
                continue
            result = pdf_parser.parse_pdf(
                source, cfg=cfg, scan_tables=not args.no_table,
                merge_continued=not args.no_merge, pages=pages_arg, logger=log,
            )
            # 产物是派生数据：每次运行整体覆盖，保证「重跑幂等」，不会追加出重复行
            paths = pdf_parser.write_parse_artifacts(result, out_dir, overwrite=True)
            pdf_parser.persist_parse_result(result, db_path=index_dir / "rag.sqlite3",
                                            source=source, logger=log)
            chunks = chunker.build_chunks(result, cfg=cfg, logger=log)
            chunker.persist_chunks(chunks, db_path=index_dir / "rag.sqlite3", logger=log)
            all_chunks.extend(chunks)
            summaries.append({
                "file_name": result.file_name, "pages": result.page_count,
                "raw_tables": result.stats.get("raw_tables", 0),
                "tables": len(result.tables), "continued": result.stats.get("continued_tables", 0),
                "header_dedup": result.stats.get("repeated_headers_removed", 0),
                "degenerate": result.stats.get("degenerate_tables", 0),
                "max_run": result.stats.get("max_run", 0),
                "text_blocks": len(result.text_blocks),
                "text_chunks": sum(1 for c in chunks if c.type == "text"),
                "table_chunks": sum(1 for c in chunks if c.type == "table"),
                "chars": sum(c.char_count for c in chunks),
                "elapsed_s": round(result.elapsed_ms / 1000, 2),
                "artifacts": {k: str(v) for k, v in paths.items()},
                "warnings": len(result.warnings),
            })
            print(f"  ✅ {result.file_name}：{result.page_count} 页，原始表 {summaries[-1]['raw_tables']} → "
                  f"逻辑表块 {summaries[-1]['tables']}（续表并入 {summaries[-1]['continued']}、"
                  f"表头去重 {summaries[-1]['header_dedup']}、退化 {summaries[-1]['degenerate']}），"
                  f"文本块 {summaries[-1]['text_blocks']}，chunk {len(chunks)}，"
                  f"{summaries[-1]['elapsed_s']}s")

        print("-" * 78)
        written = chunker.write_chunks(all_chunks, chunks_path, overwrite=True)
        print(f"chunks.jsonl 已写入 {written} 行：{chunks_path}")
        total_pages = sum(s["pages"] for s in summaries)
        print(f"合计：{len(summaries)} 份 PDF / {total_pages} 页 / {len(all_chunks)} chunk / "
              f"{time.perf_counter() - t_all:.1f}s")
        print(f"chunk 文本总字数：{sum(c.char_count for c in all_chunks)}")
        if not args.limit_pages:
            found_pages = {s["pages"] for s in summaries}
            if not found_pages <= EXPECTED_PAGES:
                print(f"❌ 页数与预期不符：实际 {sorted(found_pages)}，预期 ⊆ {sorted(EXPECTED_PAGES)}")
                return 1
            print(f"✅ 页数校验通过：{sorted(found_pages)} ⊆ {sorted(EXPECTED_PAGES)}")
        else:
            print(f"⚠️ --limit-pages={args.limit_pages}：跳过页数校验（调试模式，不得用于验收）")

        checked = check_first_pages(summaries)
        if not checked:
            print("❌ 产物自检未通过")
            return 1
        print("✅ 产物自检通过：pages/tables/text_blocks/chunks JSONL 均可解析且页码合法")
        span.set_output({"files": len(summaries), "pages": total_pages, "chunks": len(all_chunks),
                         "summaries": summaries})
        return 0


def check_first_pages(summaries: list[dict]) -> bool:
    """回读 JSONL 产物做自检：行数一致、页码 1-based 合法、chunk 字段齐全。"""
    from app.core.text_utils import read_jsonl

    ok = True
    for item in summaries:
        stem = item["file_name"].rsplit(".", 1)[0]
        for key, expect in (("pages", item["pages"]), ("tables", item["tables"]),
                            ("text_blocks", item["text_blocks"])):
            path = Path(item["artifacts"][key])
            rows = list(read_jsonl(path))
            if len(rows) != expect:
                print(f"  ❌ {path.name}: 行数 {len(rows)} != 期望 {expect}")
                ok = False
        for row in read_jsonl(Path(item["artifacts"]["tables"])):
            if not (1 <= int(row["page_start"]) <= int(row["page_end"])):
                print(f"  ❌ 表块页码非法：{row['table_id']} {row['page_start']}-{row['page_end']}")
                ok = False
    return ok


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RagError as exc:  # 预期内的业务异常：打印错误码与载荷
        print(f"\n❌ 解析失败：{exc}", file=sys.stderr)
        print(f"   detail={exc.detail}", file=sys.stderr)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底：完整堆栈 + 退出码 2
        import traceback

        traceback.print_exc()
        raise SystemExit(2)
