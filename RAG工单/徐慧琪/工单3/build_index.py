# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：建库脚本（多文档解析 → 表格结构化 → OCR 兜底 → 分块 → 向量化 → Qdrant）

与 01/02 的差异（工单3 核心）：
  1. **多文档**：默认把《招股说明书1.pdf》《招股说明书2.pdf》一起建库；
  2. **表格结构化**：表格不再整表入库，而是拆成「表头块 + 行级块」（table_chunker），
     元数据（table_id / table_html / table_header / row_index …）全部进 payload；
  3. **表格 OCR 兜底**：低质量表（MinerU 未解析出结构）所在页批量调 PaddleOCR-VL，
     结果作为该表内容入库；
  4. 表格结构化中间产物落盘 data/tables/<doc>_tables.json，供答案定位报告使用。

用法：
    python build_index.py                    # 双文档全量建库（解析缓存复用）
    python build_index.py --rebuild          # 强制重新解析
    python build_index.py --source 招股说明书2.pdf
    python build_index.py --limit-pages 20   # 冒烟：只处理前 20 页
    python build_index.py --no-embed         # 只解析+分块，不向量化
    python build_index.py --fallback-only    # 跳过 MinerU，用 PyMuPDF 快速解析
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402  —— 必须最先导入


def _banner(text: str) -> None:
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}", flush=True)


def _build_one(pdf_path: str, args) -> tuple[list[dict], dict]:
    """单文档：解析 → 表格兜底 → 分块。返回 (chunks, 统计)。"""
    from src import chunker, config, ocr_fallback, pdf_parser, table_chunker

    source = os.path.basename(pdf_path)
    _banner(f"解析 {source}")

    t0 = time.time()
    parsed = pdf_parser.parse_pdf(pdf_path, config.PARSED_DIR,
                                  reuse=not args.rebuild,
                                  force_fallback=args.fallback_only)
    items = parsed["items"]
    if args.limit_pages > 0:
        items = [it for it in items if it["page_idx"] < args.limit_pages]
    page_total = args.limit_pages or parsed["page_count"]
    print(f"解析器：{parsed['parser']} | 条目 {len(items)} | 页数 {parsed['page_count']}"
          f" | 用时 {time.time() - t0:.1f}s", flush=True)

    rep = pdf_parser.quality_report(items, page_total)
    print(f"质量报告：有文本页 {rep['pages_with_text']}/{page_total}，"
          f"表格 {rep['n_tables']}，公式 {rep['n_equations']}，图片 {rep['n_images']}，"
          f"低文本页 {len(rep['low_text_pages'])}")

    # ---- 表格结构化扫描（工单3）：找出低质量表，决定是否 OCR 兜底 ----
    t0 = time.time()
    headings = chunker.item_headings(items)     # 给无标题表格补章节标题当表名
    records, low_pages = table_chunker.analyze_tables(items, source, headings)
    n_low = sum(1 for r in records if r["low_quality"])
    print(f"表格扫描：共 {len(records)} 张表，低质量 {n_low} 张"
          f"（涉及 {len(low_pages)} 页）| 用时 {time.time() - t0:.1f}s", flush=True)

    ocr_pages: dict[int, str] = {}
    if low_pages and not args.no_ocr:
        _banner(f"表格 OCR 兜底（{source}，{len(low_pages)} 页）")
        t0 = time.time()
        from src import table_parser
        ocr_pages = table_parser.ocr_fallback_tables(
            pdf_path, low_pages, os.path.join(config.PARSED_DIR, "_ocr_pages"))
        print(f"PaddleOCR-VL 兜底：{len(ocr_pages)} 页返回文本"
              f" | 用时 {time.time() - t0:.1f}s", flush=True)
    elif low_pages:
        print(f"表格 OCR 兜底：已关闭；{len(low_pages)} 页低质量表将按原文本降级入库")

    # ---- 文本低文本页兜底（沿用 01/02：MinerU 成功时不重复 OCR）----
    if (not args.no_ocr and rep["low_text_pages"]
            and parsed["parser"] != "mineru"):
        _banner(f"PaddleOCR-VL 兜底（低文本页，{source}）")
        t0 = time.time()
        items, ocr_stat = ocr_fallback.augment_items(items, pdf_path,
                                                     config.PARSED_DIR)
        if args.limit_pages > 0:
            items = [it for it in items if it["page_idx"] < args.limit_pages]
        print(f"OCR 兜底：{ocr_stat} | 用时 {time.time() - t0:.1f}s", flush=True)

    # ---- 分块 ----
    _banner(f"分块 {source}（标题层级 + 表格结构化 + 父子块）")
    t0 = time.time()
    chunks = chunker.build_chunks(items, source, ocr_pages=ocr_pages)
    st = chunker.stats(chunks)
    print(f"chunks={st['n_chunks']} parents={st['n_parents']} "
          f"types={st['by_type']} avg={st['avg_chars']}字 max={st['max_chars']}字 "
          f"| 用时 {time.time() - t0:.1f}s", flush=True)
    print(f"表格：{st['tables']}", flush=True)

    # ---- 表格中间产物落盘（答案定位报告 / 专项说明用）----
    if records:
        from src import table_parser as table_parser_mod
        stem = os.path.splitext(source)[0]
        path = os.path.join(config.TABLE_DIR, f"{stem}_tables.json")
        table_parser_mod.save_tables(records, path)
        print(f"表格结构化产物：{path}（{len(records)} 张）", flush=True)

    return chunks, {"source": source, "parser": parsed["parser"],
                    "tables": len(records), "low_quality_tables": n_low,
                    "ocr_pages": len(ocr_pages), "stats": st}


def main() -> int:
    from src import chunker, config, embedder, vector_store

    ap = argparse.ArgumentParser(description="工单03 建库脚本（多文档 + 表格结构化）")
    ap.add_argument("--rebuild", action="store_true", help="强制重新解析（忽略缓存）")
    ap.add_argument("--fallback-only", action="store_true",
                    help="跳过 MinerU，直接用 PyMuPDF 解析（快速）")
    ap.add_argument("--limit-pages", type=int, default=0,
                    help="只处理前 N 页（0=全部）")
    ap.add_argument("--no-ocr", action="store_true", help="关闭 PaddleOCR-VL 兜底")
    ap.add_argument("--no-embed", action="store_true",
                    help="只解析+分块，不做向量化入库")
    ap.add_argument("--source", default="", help="只建某个文档（文件名或路径）")
    args = ap.parse_args()

    t_all = time.time()
    _banner(f"工单03 建库 | {config.WORKORDER_ID}")
    print(f"collection：{config.COLLECTION_OPT} | Qdrant：{config.QDRANT_MODE}")

    targets = config.SOURCE_PDFS
    if args.source:
        targets = [p for p in config.SOURCE_PDFS
                   if os.path.basename(p) == os.path.basename(args.source)]
        if not targets:
            targets = [args.source]
    print(f"待建库文档（{len(targets)}）：{[os.path.basename(p) for p in targets]}")

    all_chunks: list[dict] = []
    doc_stats: list[dict] = []
    for pdf in targets:
        if not os.path.isfile(pdf):
            print(f"[跳过] 文件不存在：{pdf}", flush=True)
            continue
        chunks, stat = _build_one(pdf, args)
        all_chunks.extend(chunks)
        doc_stats.append(stat)

    if not all_chunks:
        print("没有可用文档，退出。")
        return 1

    chunk_file = os.path.join(config.CHUNK_DIR, "chunks.json")
    os.makedirs(config.CHUNK_DIR, exist_ok=True)
    with open(chunk_file, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False)
    st = chunker.stats(all_chunks)
    _banner("汇总")
    print(f"总 chunks={st['n_chunks']} parents={st['n_parents']}")
    print(f"按类型：{st['by_type']}")
    print(f"按文档：{st['by_source']}")
    print(f"表格：{st['tables']}")
    print(f"已写入 {chunk_file}")

    if args.no_embed:
        _banner(f"完成（跳过向量化）总用时 {time.time() - t_all:.1f}s")
        return 0

    # ---- 向量化 ----
    _banner("向量化（bge-m3）")
    t0 = time.time()
    vectors = embedder.embed_texts([c["text"] for c in all_chunks],
                                   show_progress=True)
    print(f"向量 {len(vectors)} 条（dim={embedder.get_dim()}）"
          f" | 用时 {time.time() - t0:.1f}s", flush=True)

    # ---- 入库 ----
    _banner("写入 Qdrant")
    t0 = time.time()
    vector_store.ensure_collection(dim=embedder.get_dim(), recreate=True)
    n = vector_store.upsert_chunks(all_chunks, vectors)
    print(f"入库 {n} 点 | 用时 {time.time() - t0:.1f}s", flush=True)

    _banner(f"建库完成：{vector_store.info()} | 总用时 {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
