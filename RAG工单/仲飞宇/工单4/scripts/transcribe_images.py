#!/usr/bin/env python
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单04 - 图像内容解析及检索优化
"""
图像转写作业台（工单04）。**不碰 Milvus** —— 只做解析 + 渲染 + VLM + 缓存，
所以可以反复跑，改提示词/核对/重跑都不影响已入库的数据。

    # 只解析、渲染、查缓存，不调 VLM —— 最快，看"有哪些图、缓存命中多少"
    $PY scripts/transcribe_images.py --scan

    # 真跑 VLM（缓存命中的会跳过）
    $PY scripts/transcribe_images.py --run
    $PY scripts/transcribe_images.py --run --doc liyuan --pages 71,38

    # 核对与修正
    $PY scripts/transcribe_images.py --list            # 一览：页/类型/题注/是否已核对
    $PY scripts/transcribe_images.py --show 71         # 完整 JSON + PNG 路径
    $PY scripts/transcribe_images.py --dump-png 71 /tmp/x.png   # 把图导出来肉眼看
    $PY scripts/transcribe_images.py --review 71       # 编辑 reviewed_text（人工覆写）
    $PY scripts/transcribe_images.py --refresh --pages 71   # 强制重跑这张的 VLM

    # 统计
    $PY scripts/transcribe_images.py --stats

【为什么要 --dump-png】本机没装 poppler，`Read` 工具打不开 PDF。
要把图拿来给我或人肉核对，只能靠这个命令先把图区渲染成 PNG。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings                                     # noqa: E402
from app.core import image_cache, vision                            # noqa: E402
from app.core.doc_profiles import DOC_PROFILES                      # noqa: E402
from app.core.pdf_parser import parse_pdf_full                      # noqa: E402


def _pdf(key: str) -> Path:
    return settings.data_path / "raw" / DOC_PROFILES[key].doc_name


def _figures(key: str, limit: int | None = None):
    """解析一份 PDF 并返回 (figures, stats)。"""
    pages, st = parse_pdf_full(_pdf(key), limit=limit)
    return [f for pc in pages for f in pc.figures], st


def _key_for(fig, doc) -> str:
    import asyncio as _a
    png = vision.render_clip(doc, fig.page_no, fig.bbox, settings.image_render_dpi)
    return image_cache.cache_key(png, settings.vlm_model)


def cmd_scan(keys, limit) -> int:
    import pymupdf
    print(f"{'文档':<20}{'页':>6}{'类型':<9}{'题注':<34}{'缓存'}")
    total = hit = 0
    for k in keys:
        figs, st = _figures(k, limit)
        doc = pymupdf.open(_pdf(k))
        try:
            for f in figs:
                total += 1
                ck = _key_for(f, doc)
                rec = image_cache.load(ck)
                mark = "✓已转写" if rec and rec.vlm_text else ("✔人工核对" if rec and rec.reviewed else "· 未转写")
                hit += 1 if rec else 0
                print(f"{k:<20}{f.page_label:>6}{f.kind:<9}{f.title[:32]:<34}{mark}")
        finally:
            doc.close()
    print(f"\n合计图区 {total}，缓存命中 {hit}")
    return 0


def cmd_run(keys, pages_filter, limit, force) -> int:
    import pymupdf
    from app.core.pdf_parser import parse_pdf_full as _parse

    async def main() -> int:
        stats_all = vision.FigureStats()
        for k in keys:
            path = _pdf(k)
            figs_pre, st = _figures(k, limit)
            if pages_filter:
                keep = {int(p) - 1 for p in pages_filter}      # 印刷页 → index（偏移 0）
                figs_pre = [f for f in figs_pre if f.page_no in keep]
            if not figs_pre:
                print(f"[{k}] 没有匹配的图区")
                continue
            pages, _ = _parse(path, limit=limit)               # 重新解析拿 PageContent
            want = {(f.page_no, f.bbox, f.kind) for f in figs_pre}
            for pc in pages:
                pc.figures = [f for f in pc.figures
                              if (f.page_no, f.bbox, f.kind) in want]
            print(f"[{k}] 待处理图区 {sum(len(pc.figures) for pc in pages)}")

            def prog(i, n, fig):
                print(f"    {i}/{n} 页 {fig.page_label} {fig.kind} "
                      f"{'（缓存命中）' if not fig.text else ''}", flush=True)

            tr = vision.FigureTranscriber(path, progress=prog, force=force)
            st2 = await tr.run(pages)
            print(f"    完成：VLM 调用 {st2.n_vlm_calls}｜缓存命中 {st2.n_cache_hits}"
                  f"｜失败 {st2.n_failures}｜{st2.seconds:.0f}s")
            stats_all.n_figures += st2.n_figures
            stats_all.n_vlm_calls += st2.n_vlm_calls
            stats_all.n_cache_hits += st2.n_cache_hits
            stats_all.n_failures += st2.n_failures
        print("\n=== 合计 ===")
        for k, v in stats_all.as_dict().items():
            print(f"  {k}: {v}")
        return 0

    # 转写前把聊天模型赶下显存（8GB 卡装不下两者，见 vision 的注释）
    from app.core.ollama_client import get_client
    ok = asyncio.run(vision.unload_chat_model(get_client()))
    print(f"[显存] 卸载聊天模型 {'成功' if ok else '失败（不阻断）'}")
    try:
        return asyncio.run(main())
    finally:
        asyncio.run(vision.unload_vlm(get_client()))


def cmd_list() -> int:
    recs = image_cache.all_transcripts()
    if not recs:
        print("缓存为空。先跑 --run")
        return 0
    print(f"{'页':>6}{'文档':<18}{'类型':<9}{'核对':<6}{'秒':>6}  题注 / 转写首行")
    for r in sorted(recs, key=lambda x: (x.doc_name, x.page_no)):
        first = (r.effective_text or "").splitlines()
        head = first[0][:44] if first else "（空）"
        print(f"{r.page_label:>6}{r.doc_name[:16]:<18}{r.kind:<9}"
              f"{'✔' if r.reviewed else '□':<6}{r.seconds:>6.1f}  {r.caption[:28]} ｜ {head}")
    print(f"\n共 {len(recs)} 条，已人工核对 {sum(1 for r in recs if r.reviewed)} 条")
    return 0


def _pick(page: str) -> image_cache.Transcript | None:
    for r in image_cache.all_transcripts():
        if r.page_label == page or str(r.page_no) == page:
            return r
    return None


def cmd_show(page: str) -> int:
    r = _pick(page)
    if r is None:
        print(f"缓存里没有页 {page} 的记录")
        return 1
    print(image_cache.transcript_path(r.key).read_text(encoding="utf-8"))
    print("PNG:", image_cache.png_path(r.key))
    return 0


def cmd_dump_png(page: str, dest: str) -> int:
    r = _pick(page)
    if r is None:
        print(f"缓存里没有页 {page} 的记录")
        return 1
    src = image_cache.png_path(r.key)
    if not src.exists():
        print(f"PNG 不在：{src}")
        return 1
    shutil.copy(src, dest)
    print(f"已导出 {src} → {dest}（{src.stat().st_size // 1024} KB）")
    return 0


def cmd_review(page: str) -> int:
    r = _pick(page)
    if r is None:
        print(f"缓存里没有页 {page} 的记录")
        return 1
    p = image_cache.transcript_path(r.key)
    editor = os.environ.get("EDITOR") or "nano"
    print(f"编辑 {p}（改 reviewed_text 字段，保存后生效）")
    subprocess.call([editor, str(p)])
    r2 = image_cache.load(r.key)
    print(f"重新载入：reviewed={r2.reviewed} 有效文本长度={len(r2.effective_text)}")
    return 0


def cmd_stats(keys) -> int:
    recs = image_cache.all_transcripts()
    print("=== 缓存 ===")
    print(f"  记录数 {len(recs)}｜已核对 {sum(1 for r in recs if r.reviewed)}"
          f"｜VLM 成功 {sum(1 for r in recs if r.vlm_text)}"
          f"｜失败 {sum(1 for r in recs if r.note and not r.vlm_text)}")
    by_doc: dict[str, int] = {}
    for r in recs:
        by_doc[r.doc_name] = by_doc.get(r.doc_name, 0) + 1
    for k, v in sorted(by_doc.items()):
        print(f"    {k}: {v}")
    print("\n=== 图区定位 ===")
    for k in keys:
        figs, st = _figures(k)
        print(f"  {DOC_PROFILES[k].doc_name}: 图区 {st.n_figures}"
              f"（位图 {st.n_figures_bitmap} / 矢量 {st.n_figures_vector}）"
              f"｜滤掉噪声位图实例 {st.n_images_dropped}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="工单04 · 图像转写作业台")
    ap.add_argument("--doc", default="all",
                    help=f"哪份文档：{'/'.join(DOC_PROFILES)} 或 all")
    ap.add_argument("--pages", default=None,
                    help="只处理这些印刷页（逗号分隔），如 71,38")
    ap.add_argument("--limit", type=int, default=None, help="只解析前 N 页")
    ap.add_argument("--scan", action="store_true", help="只解析+查缓存，不调 VLM")
    ap.add_argument("--run", action="store_true", help="跑 VLM 转写")
    ap.add_argument("--list", action="store_true", help="列出缓存的转写记录")
    ap.add_argument("--show", default=None, metavar="页")
    ap.add_argument("--dump-png", nargs=2, default=None, metavar=("页", "目标路径"))
    ap.add_argument("--review", default=None, metavar="页")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存强制重跑")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    keys = list(DOC_PROFILES) if args.doc == "all" else [args.doc]
    if args.doc != "all" and args.doc not in DOC_PROFILES:
        print(f"未知文档 {args.doc}", file=sys.stderr)
        return 2
    pages_filter = [p.strip() for p in args.pages.split(",")] if args.pages else None

    if args.scan:
        return cmd_scan(keys, args.limit)
    if args.show:
        return cmd_show(args.show)
    if args.dump_png:
        return cmd_dump_png(args.dump_png[0], args.dump_png[1])
    if args.review:
        return cmd_review(args.review)
    if args.list:
        return cmd_list()
    if args.stats:
        return cmd_stats(keys)
    if args.run or args.refresh:
        return cmd_run(keys, pages_filter, args.limit, force=args.refresh)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
