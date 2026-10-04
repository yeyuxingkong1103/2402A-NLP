# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
命令行入库。

  python scripts/ingest.py                 # 全量入库
  python scripts/ingest.py --rebuild       # 先删 collection 再入库
  python scripts/ingest.py --limit-pages 40  # 只解析前 40 页（快速验证）
  python scripts/ingest.py --dry-run       # 只解析+分块，不写向量库

【为什么要有 --dry-run】Milvus 没起来时，解析/分块这两步（占全流程 80%
的调试价值）仍然可以验证。先看解析质量，再决定要不要动向量库。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="工单01 · 命令行入库")
    ap.add_argument("--pdf", default=None, help="PDF 路径，默认 data/raw/招股说明书1.pdf")
    ap.add_argument("--rebuild", action="store_true", help="先删除 collection")
    ap.add_argument("--limit-pages", type=int, default=None, help="只解析前 N 页")
    ap.add_argument("--dry-run", action="store_true", help="只解析分块，不写向量库")
    args = ap.parse_args()

    pdf = Path(args.pdf) if args.pdf else settings.data_path / "raw" / "招股说明书1.pdf"
    if not pdf.exists():
        print(f"[错误] PDF 不存在：{pdf}", file=sys.stderr)
        return 2

    print(f"PDF：{pdf}")
    print(f"页码偏移：{settings.page_label_offset}")

    if args.dry_run:
        return _dry_run(pdf, args.limit_pages)

    import asyncio

    from app.core.pipeline import IngestPipeline

    pipe = IngestPipeline()
    seen: list[str] = []

    async def run():
        task = asyncio.create_task(
            pipe.run(pdf, rebuild=args.rebuild, limit_pages=args.limit_pages))
        while not task.done():
            st = pipe.status()
            if not seen or st.stage != seen[-1]:
                seen.append(st.stage)
                print(f"  [{st.stage}] {st.message}", flush=True)
            await asyncio.sleep(1.5)
        return await task

    st = asyncio.run(run())

    print()
    if st.error:
        print(f"[失败] {st.error}")
        return 1
    for k, v in st.stats.items():
        print(f"  {k}: {v}")
    print(f"\n[完成] {st.message}")
    return 0


def _dry_run(pdf: Path, limit: int | None) -> int:
    """只解析 + 分块 + 去重，便于快速检查解析质量。"""
    from app.core.chunker import chunk_pages
    from app.core.dedup import dedup_chunks
    from app.core.pdf_parser import parse_pdf_full

    t0 = time.perf_counter()
    pages, stats = parse_pdf_full(pdf, offset=settings.page_label_offset, limit=limit)
    print(f"\n解析：{stats.n_pages} 页，耗时 {time.perf_counter() - t0:.1f}s")
    print(f"  表格 {stats.n_tables} 张｜文本块 {stats.n_text_blocks} 个")
    print(f"  剔除页眉 {stats.n_header_removed}、页脚 {stats.n_footer_removed}")
    print(f"  表格内重复文本剔除 {stats.n_table_text_dropped} 块")

    t1 = time.perf_counter()
    chunks = list(chunk_pages(pages))
    from collections import Counter
    cnt = Counter(c.chunk_type for c in chunks)
    n_text = cnt.get("text", 0)
    print(f"\n分块：{len(chunks)} 个（text {n_text} / table {cnt.get('table', 0)}"
          f" / image {cnt.get('image', 0)}），"
          f"耗时 {time.perf_counter() - t1:.1f}s")
    if n_text:
        ls = sorted(c.n_chars for c in chunks if c.chunk_type == "text")
        print(f"  字数 min={ls[0]} 中位={ls[len(ls) // 2]} max={ls[-1]}")

    t2 = time.perf_counter()
    dres = dedup_chunks(chunks)
    print(f"\n去重：保留 {len(dres.kept)}，丢弃 {len(dres.dropped)}"
          f"（精确 {dres.n_exact} / 近似 {dres.n_near}，"
          f"率 {dres.drop_rate:.1%}），耗时 {time.perf_counter() - t2:.1f}s")

    # 关键页抽查 —— 这 5 页决定 10 题成败
    print("\n关键页抽查：")
    key = {21: "法定代表人/注册资本", 29: "募集资金投资项目表",
           128: "军用收入金句", 151: "行业上下游", 152: "上下游图示"}
    for pno, why in key.items():
        if pno >= len(pages):
            continue
        pc = pages[pno]
        print(f"  p{pno}（{pc.page_label}）{why}："
              f"文本块 {len(pc.texts)}，表 {len(pc.tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
