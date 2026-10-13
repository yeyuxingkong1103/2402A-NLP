# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：建库脚本（上游解析产物 → 图像块增强 → 向量化 → Qdrant v6 → 全文索引）

与上游（工单01~05）建库的差异
------------------------------
1. **直接复用上游分块产物**：招股书解析（MinerU + 表格结构化 + 组织图矢量解析）
   已由工单 03/05 完成并落盘 data/chunks/chunks.json。本工单是"混合检索"任务，
   不重复做 PDF 解析（需求原文：「不是重新做 PDF 解析」）。需要重新解析时才加 --rebuild。
2. **图像块增强（本工单新增）**：上游 04 的 162 条 VLM 图像结构化描述随工程落盘，
   建库时把占位图像块（text="[图片]"）替换为「一句话概括 + 全文描述」，
   并带 image_id / image_path —— 这是 id5（组织结构图）、id6（IC 市场增长图）
   两道图像题可检索的前提（src/image_blocks.py）。
3. **多嵌入模型**：`--embedding-model` 选择嵌入模型（bge-m3 / bge-m3-ft-v1 / miniLM-L6），
   各自写入**独立 collection**（zhaogu_v6_hybrid[__模型名]）。
4. **三套索引同源构建**（工单核心"混合检索"的基础设施）：
   · 稠密向量索引（Qdrant）
   · 全文倒排索引（自研，src/fulltext.py，持久化 data/index/fulltext.pkl）
   · 稀疏 BM25 索引（rank_bm25，懒加载；对照臂用）

用法
----
    python build_index.py                         # 默认：复用解析产物全量建库
    python build_index.py --rebuild               # 强制重新解析 PDF（MinerU，很慢）
    python build_index.py --embedding-model bge-m3-ft-v1
    python build_index.py --no-embed              # 只生成/增强 chunks，不入库
    python build_index.py --skip-orgchart         # 不补入组织结构图块
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
    """单文档：解析 → 表格兜底 → 分块（沿用上游 03 的流水线，仅在 --rebuild 时走）。"""
    from src import chunker, config, ocr_fallback, pdf_parser, table_chunker

    source = os.path.basename(pdf_path)
    _banner(f"解析 {source}")

    t0 = time.time()
    parsed = pdf_parser.parse_pdf(pdf_path, config.PARSED_DIR,
                                  reuse=not args.rebuild,
                                  force_fallback=args.fallback_only)
    items = parsed["items"]
    page_total = parsed["page_count"]
    print(f"解析器：{parsed['parser']} | 条目 {len(items)} | 页数 {page_total}"
          f" | 用时 {time.time() - t0:.1f}s", flush=True)

    rep = pdf_parser.quality_report(items, page_total)
    print(f"质量报告：有文本页 {rep['pages_with_text']}/{page_total}，"
          f"表格 {rep['n_tables']}，公式 {rep['n_equations']}，图片 {rep['n_images']}，"
          f"低文本页 {len(rep['low_text_pages'])}")

    t0 = time.time()
    headings = chunker.item_headings(items)
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

    if (not args.no_ocr and rep["low_text_pages"]
            and parsed["parser"] != "mineru"):
        _banner(f"PaddleOCR-VL 兜底（低文本页，{source}）")
        items, ocr_stat = ocr_fallback.augment_items(items, pdf_path,
                                                     config.PARSED_DIR)
        print(f"OCR 兜底：{ocr_stat} | 用时 {time.time() - t0:.1f}s", flush=True)

    _banner(f"分块 {source}（标题层级 + 表格结构化 + 父子块）")
    t0 = time.time()
    chunks = chunker.build_chunks(items, source, ocr_pages=ocr_pages)
    st = chunker.stats(chunks)
    print(f"chunks={st['n_chunks']} parents={st['n_parents']} "
          f"types={st['by_type']} avg={st['avg_chars']}字 max={st['max_chars']}字 "
          f"| 用时 {time.time() - t0:.1f}s", flush=True)

    if records:
        from src import table_parser as table_parser_mod
        stem = os.path.splitext(source)[0]
        path = os.path.join(config.TABLE_DIR, f"{stem}_tables.json")
        table_parser_mod.save_tables(records, path)
        print(f"表格结构化产物：{path}（{len(records)} 张）", flush=True)

    return chunks, {"source": source, "parser": parsed["parser"],
                    "tables": len(records), "low_quality_tables": n_low,
                    "ocr_pages": len(ocr_pages), "stats": st}


def _load_chunks(args) -> tuple[list[dict], list[dict]]:
    """取分块结果：默认复用上游产物；--rebuild 时重新解析。"""
    from src import chunker, config

    chunk_file = os.path.join(config.CHUNK_DIR, "chunks.json")
    if not args.rebuild and os.path.isfile(chunk_file):
        with open(chunk_file, encoding="utf-8") as fh:
            chunks = json.load(fh)
        st = chunker.stats(chunks)
        _banner(f"复用上游分块产物 {chunk_file}")
        print(f"chunks={st['n_chunks']} parents={st['n_parents']} "
              f"types={st['by_type']} by_source={st['by_source']}", flush=True)
        return chunks, []

    all_chunks: list[dict] = []
    doc_stats: list[dict] = []
    for pdf in config.SOURCE_PDFS:
        if not os.path.isfile(pdf):
            print(f"[跳过] 文件不存在：{pdf}", flush=True)
            continue
        chunks, stat = _build_one(pdf, args)
        all_chunks.extend(chunks)
        doc_stats.append(stat)
    return all_chunks, doc_stats


def _restore_orgchart() -> bool:
    """把组织结构图块补入 collection（recreate 会冲掉，故每轮重建后补一次）。

    上游工单05 用矢量几何路线把招股2 第 39 页组织结构图解析成 30 个块
    （1 整树 + 29 节点），是 id5「销售部有几个部门/大客户销售部几个销售处」的证据。
    补入失败只告警不中断：主索引仍然有效。
    """
    _banner("补入组织结构图块（工单05 成果复用）")
    from src import config

    try:
        import build_orgchart
        rc = build_orgchart.ingest_orgchart(rebuild=True, skip_ocr=True)
        reason = ""
    except Exception as exc:  # noqa: BLE001 —— 补入失败不得拖垮主索引建库
        rc, reason = 1, repr(exc)

    if rc != 0:
        detail = f"（退出码 {rc}，原因 {reason}）" if reason else f"（退出码 {rc}）"
        print(f"\n[警告] 组织结构图块补入失败{detail}：\n"
              f"       id5「组织结构图」将失去证据块；主索引不受影响。\n"
              f"       手动恢复：python build_orgchart.py --rebuild\n",
              file=sys.stderr, flush=True)
        return False

    print(f"组织结构图块已补入（{config.ORGCHART_SOURCE} 第 "
          f"{config.ORGCHART_PAGE_IDX + 1} 页）", flush=True)
    return True


def _embed_and_upsert(all_chunks: list[dict], rebuild_orgchart: bool = True) -> int:
    """向量化 → recreate 写入当前活动模型的 collection → 补回组织图块。"""
    from src import embedder, vector_store

    model = embedder.active_embedding_model()
    _banner(f"向量化（{model}）")
    t0 = time.time()
    vectors = embedder.embed_texts([c["text"] for c in all_chunks],
                                   show_progress=True)
    dim = embedder.get_dim()
    print(f"向量 {len(vectors)} 条（dim={dim}）| 用时 {time.time() - t0:.1f}s",
          flush=True)

    _banner(f"写入 Qdrant（collection={vector_store.default_name()}）")
    t0 = time.time()
    vector_store.ensure_collection(dim=dim, recreate=True)
    n = vector_store.upsert_chunks(all_chunks, vectors)
    print(f"入库 {n} 点 | 用时 {time.time() - t0:.1f}s", flush=True)

    if rebuild_orgchart:
        _restore_orgchart()
    return n


def main() -> int:
    from src import chunker, config, embedder, vector_store

    ap = argparse.ArgumentParser(
        description="工单06 建库脚本（复用上游解析 + 图像块增强 + 多嵌入模型）")
    ap.add_argument("--rebuild", action="store_true",
                    help="强制重新解析 PDF（忽略上游分块产物，慢）")
    ap.add_argument("--fallback-only", action="store_true",
                    help="跳过 MinerU，直接用 PyMuPDF 解析（仅 --rebuild 时有意义）")
    ap.add_argument("--no-ocr", action="store_true", help="关闭 PaddleOCR-VL 兜底")
    ap.add_argument("--no-embed", action="store_true",
                    help="只生成/增强 chunks 与全文索引，不做向量化入库")
    ap.add_argument("--embedding-model", default=config.EMBEDDING_MODEL_NAME,
                    help=f"嵌入模型：{list(config.EMBEDDING_MODELS)}")
    ap.add_argument("--skip-orgchart", action="store_true",
                    help="不补入组织结构图块")
    args = ap.parse_args()

    t_all = time.time()
    _banner(f"工单06 建库 | {config.WORKORDER_ID}")

    embedder.set_active_model(args.embedding_model)
    print(f"嵌入模型：{args.embedding_model} | "
          f"collection：{config.collection_name(args.embedding_model)} | "
          f"Qdrant：{config.QDRANT_MODE}", flush=True)

    # ---- 1) 分块（复用上游产物或重新解析）----
    all_chunks, _doc_stats = _load_chunks(args)
    if not all_chunks:
        print("没有可用文档，退出。")
        return 1

    # ---- 2) 图像块增强（本工单新增：把 VLM 描述注入图像块）----
    from src import image_blocks
    _banner("图像块增强（复用上游工单04 的 VLM 结构化描述）")
    all_chunks, img_stat = image_blocks.enrich_image_chunks(all_chunks)
    print(f"图像块 {img_stat['n_images']} 个：命中描述 {img_stat['matched']}，"
          f"未命中 {img_stat['unmatched']}，增强后块数 {img_stat['n_new_blocks']}",
          flush=True)

    chunk_file = os.path.join(config.CHUNK_DIR, "chunks.json")
    os.makedirs(config.CHUNK_DIR, exist_ok=True)
    with open(chunk_file, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False)
    st = chunker.stats(all_chunks)
    _banner("汇总")
    print(f"总 chunks={st['n_chunks']} parents={st['n_parents']}")
    print(f"按类型：{st['by_type']}")
    print(f"按文档：{st['by_source']}")
    print(f"已写入 {chunk_file}", flush=True)

    if args.no_embed:
        _banner(f"完成（跳过向量化）总用时 {time.time() - t_all:.1f}s")
        return 0

    # ---- 3) 向量化 + 入库 ----
    _embed_and_upsert(all_chunks, rebuild_orgchart=not args.skip_orgchart)

    # ---- 4) 全文倒排索引（自研）+ BM25 稀疏索引（对照臂）----
    _banner("构建全文倒排索引（自研）")
    from src import fulltext, retriever
    t0 = time.time()
    idx = fulltext.refresh()
    print(f"倒排索引：{idx.stats()} | 用时 {time.time() - t0:.1f}s", flush=True)
    print("构建 BM25 稀疏索引（对照臂）……", flush=True)
    retriever.refresh()

    _banner(f"建库完成：{vector_store.info()} | 总用时 {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
