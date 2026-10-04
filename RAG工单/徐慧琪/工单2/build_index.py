# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：建库脚本（解析 → OCR 兜底 → 分块 → 向量化 → Qdrant 入库）

用法：
    python build_index.py                 # 全量建库（解析缓存复用）
    python build_index.py --rebuild       # 强制重新解析
    python build_index.py --limit-pages 20  # 只处理前 20 页（冒烟）
    python build_index.py --no-embed      # 只解析+分块，不做向量化（省 GPU 时间）
    python build_index.py --fallback-only # 跳过 MinerU，直接用 PyMuPDF（快速全量）

产物：
    data/parsed/<stem>/auto/<stem>.md + _content_list.json（MinerU）
    data/chunks/chunks.json（分块结果，含父子块与元数据）
    Qdrant collection: config.COLLECTION_OPT（本地模式落盘 data/qdrant/）
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


def main() -> int:
    from src import (chunker, config, embedder, ocr_fallback, pdf_parser,
                     vector_store)

    ap = argparse.ArgumentParser(description="工单02 建库脚本")
    ap.add_argument("--rebuild", action="store_true", help="强制重新解析（忽略缓存）")
    ap.add_argument("--fallback-only", action="store_true",
                    help="跳过 MinerU，直接用 PyMuPDF 解析（快速）")
    ap.add_argument("--limit-pages", type=int, default=0,
                    help="只处理前 N 页（0=全部）")
    ap.add_argument("--no-ocr", action="store_true", help="关闭 PaddleOCR-VL 兜底")
    ap.add_argument("--no-embed", action="store_true",
                    help="只解析+分块，不做向量化入库")
    ap.add_argument("--source", default=config.SOURCE_PDF, help="待解析 PDF 路径")
    args = ap.parse_args()

    t_all = time.time()
    _banner(f"工单02 建库 | {config.WORKORDER_ID}")
    print(f"源文件：{args.source}")

    # ---- 1) 解析 ----
    _banner("1/5 解析 PDF")
    t0 = time.time()
    parsed = pdf_parser.parse_pdf(config.SOURCE_PDF if args.source == config.SOURCE_PDF
                                  else args.source,
                                  config.PARSED_DIR,
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

    # ---- 2) OCR 兜底 ----
    # MinerU 自带 OCR（models/OCR/paddleocr_torch）并对扫描页/图片页做了识别，
    # 因此 MinerU 成功解析时无需外部 PaddleOCR-VL 兜底（否则会把已识别的页
    # 再跑一遍 OCR，实测浪费 10 分钟且无增益）。
    if not args.no_ocr and rep["low_text_pages"] and parsed["parser"] != "mineru":
        _banner("2/5 PaddleOCR-VL 兜底（低文本页）")
        t0 = time.time()
        items, ocr_stat = ocr_fallback.augment_items(parsed["items"], args.source,
                                                     config.PARSED_DIR)
        if args.limit_pages > 0:
            items = [it for it in items if it["page_idx"] < args.limit_pages]
        print(f"OCR 兜底：{ocr_stat} | 用时 {time.time() - t0:.1f}s", flush=True)
    else:
        print("2/5 OCR 兜底：跳过（关闭或无低文本页）")

    # ---- 3) 分块 ----
    _banner("3/5 分块（标题层级 + 父子块）")
    t0 = time.time()
    chunks = chunker.build_chunks(items, parsed["source"])
    st = chunker.stats(chunks)
    print(f"chunks={st['n_chunks']} parents={st['n_parents']} "
          f"types={st['by_type']} avg={st['avg_chars']}字 max={st['max_chars']}字 "
          f"| 用时 {time.time() - t0:.1f}s", flush=True)
    os.makedirs(config.CHUNK_DIR, exist_ok=True)
    chunk_file = os.path.join(config.CHUNK_DIR, "chunks.json")
    with open(chunk_file, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False)
    print(f"已写入 {chunk_file}")

    if args.no_embed:
        _banner(f"完成（跳过向量化）总用时 {time.time() - t_all:.1f}s")
        return 0

    # ---- 4) 向量化 ----
    _banner("4/5 向量化（bge-m3）")
    t0 = time.time()
    vectors = embedder.embed_texts([c["text"] for c in chunks], show_progress=True)
    print(f"向量 {len(vectors)} 条（dim={embedder.get_dim()}）"
          f" | 用时 {time.time() - t0:.1f}s", flush=True)

    # ---- 5) 入库 ----
    _banner("5/5 写入 Qdrant")
    t0 = time.time()
    vector_store.ensure_collection(dim=embedder.get_dim(), recreate=True)
    n = vector_store.upsert_chunks(chunks, vectors)
    print(f"入库 {n} 点 | 用时 {time.time() - t0:.1f}s", flush=True)

    _banner(f"建库完成：{vector_store.info()} | 总用时 {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
