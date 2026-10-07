"""
索引构建脚本（工单04：多文档 + 表格解析 + 图像解析）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

用法：
    python scripts/build_index.py                      # 主线索引（表格 + 图像都开）
    python scripts/build_index.py --out noimage        # 关掉图像解析（**工单04 的对照组**）
    python scripts/build_index.py --no-tables          # 关掉表格解析（工单03 的对照组）
    python scripts/build_index.py --dry-run            # 只解析 + 分块，不向量化不入库
    python scripts/build_index.py --only liyuan        # 只建其中一份（调试用）

--out noimage 就是工单04 要对比的「优化前」：图**不经过多模态模型**，
图内文字与图题照旧留在正文里当普通文本行（也就是工单1/2/3 的做法）。
两份索引的解析/分块/检索算法完全一致，只有「图有没有被多模态解析」这一个变量不同。

干跑（--dry-run）必须看两个数：平均块长 **和最长块长**。
只看平均会漏掉长尾巨块 —— 「平均正常 + 最长爆表」才是有问题的信号。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chunker import chunk_document  # noqa: E402
from src.pdf_parser import parse_pdf  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO_IMAGE


def resolve_docs(only: str) -> list[dict]:
    docs = config.DOCS
    if only:
        docs = [d for d in docs if d["key"] == only]
        if not docs:
            raise SystemExit(f"[FAIL] config.DOCS 里没有 key={only}")
    return docs


def load_figure_assets(doc_key: str, with_images: bool
                       ) -> tuple[list[dict], dict[int, list[str]]]:
    """读某份文档的图清单与正文剔除表。

    with_images=False（工单04 对照组）时不传 figures、
    也**不传 figure_drops** —— 图内散字和图题照旧留在正文里，
    这正是"图没被解析"时系统的真实形态。只传 figures 不传 drops 会变成
    "图内容既在正文里、又有一条图像块"，不是干净的对照。
    """
    if not with_images:
        return [], {}
    if not config.FIGURES_JSON.exists():
        print("      [warn] 未找到 figures.json，图像解析将不生效（先跑 scripts/build_images.py）")
        return [], {}
    figs = json.loads(config.FIGURES_JSON.read_text(encoding="utf-8"))
    figs = [f for f in figs if f["doc_key"] == doc_key]
    miss = sum(1 for f in figs if not f.get("desc"))
    if miss:
        print(f"      [warn] {doc_key} 有 {miss}/{len(figs)} 张图没有语义描述"
              f"（先跑 scripts/parse_images.py）")

    drop_file = config.PROCESSED_DIR / f"figure_drop_{doc_key}.json"
    drops: dict[int, list[str]] = {}
    if drop_file.exists():
        raw = json.loads(drop_file.read_text(encoding="utf-8"))
        drops = {int(k): v for k, v in raw.items()}
    return figs, drops


def main() -> int:
    ap = argparse.ArgumentParser(description=f"构建索引 | 工单：{WORK_ORDER_NO}")
    ap.add_argument("--no-tables", action="store_true",
                    help="跳过表格解析（工单03 的「优化前」对照组）")
    ap.add_argument("--no-images", action="store_true",
                    help="跳过图像解析（工单04 的「优化前」对照组）")
    ap.add_argument("--dry-run", action="store_true", help="只解析与分块，不向量化不入库")
    ap.add_argument("--only", default="", help="只建指定 key 的文档（xingtu / liyuan）")
    ap.add_argument("--out", default="", choices=["", "notable", "noimage"],
                    help="notable/noimage = 落到对应的对照组索引目录")
    args = ap.parse_args()

    config.ensure_dirs()
    docs = resolve_docs(args.only)
    with_tables = not args.no_tables
    with_images = not args.no_images

    print(f"[工单] {WORK_ORDER_NO}")
    print(f"[语料] {len(docs)} 份文档；表格解析：{'开' if with_tables else '关'}；"
          f"图像解析（多模态）：{'开' if with_images else '关（对照组）'}\n")

    all_chunks = []
    all_tables = 0
    n_figs_used = 0
    source_paths: list[Path] = []
    seq = 0

    for di, d in enumerate(docs, 1):
        pdf = config.RAW_DIR / d["file"]
        if not pdf.exists():
            print(f"[FAIL] 文档不存在：{pdf}")
            return 2
        source_paths.append(pdf)

        print(f"[{di}/{len(docs)}] 解析 {pdf.name} —— {d['name']}")
        figs, drops = load_figure_assets(d["key"], with_images)
        n_figs_used += sum(1 for f in figs if f.get("desc"))

        t0 = time.perf_counter()
        parsed = parse_pdf(
            pdf,
            with_tables=with_tables,
            header_res=[re.compile(d["header_re"])],   # 页眉按文档给，避免漏剔另一家的页眉
            doc_name=d["name"],
            figure_drops=drops,
            figures=figs,
        )
        parse_s = time.perf_counter() - t0
        empty = sum(1 for p in parsed.pages if not p.text.strip())
        tbl_pages = len({t.page for t in parsed.tables})
        n_tables = len(parsed.tables)
        all_tables += n_tables
        print(f"      {parsed.total_pages} 页 / 耗时 {parse_s:.1f}s；"
              f"空文本页 {empty}；结构化表格 {n_tables} 张（分布在 {tbl_pages} 页）；"
              f"图 {len(parsed.figures)} 个（有描述 {sum(1 for f in parsed.figures if f.get('desc'))}）")

        t0 = time.perf_counter()
        chunks = chunk_document(parsed, doc_key=d["key"], seq_start=seq)
        seq += len(chunks)
        lens = [len(c.text) for c in chunks] or [0]
        n_tbl_chunks = sum(1 for c in chunks if c.type == "table")
        n_img_chunks = sum(1 for c in chunks if c.type == "image")
        print(f"      分块 {len(chunks)} 条（表格块 {n_tbl_chunks} / 图像块 {n_img_chunks}），"
              f"块长 平均 {sum(lens)/len(lens):.0f} / 最短 {min(lens)} / 最长 {max(lens)}"
              f"（上限 {config.CHUNK_MAX_LENGTH}），耗时 {time.perf_counter()-t0:.1f}s")
        if max(lens) > config.CHUNK_MAX_LENGTH:
            print("[FAIL] 存在超上限的块，分块兜底失效！")
            return 3
        all_chunks.extend(chunks)

    lens = [len(c.text) for c in all_chunks] or [0]
    print(f"\n[合计] {len(all_chunks)} 块"
          f"（表格块 {sum(1 for c in all_chunks if c.type=='table')}"
          f" / 图像块 {sum(1 for c in all_chunks if c.type=='image')}），"
          f"表格 {all_tables} 张，图 {n_figs_used} 个")
    print(f"       块长 平均 {sum(lens)/len(lens):.0f} / 最短 {min(lens)} / 最长 {max(lens)}")
    per_doc: dict[str, int] = {}
    for c in all_chunks:
        per_doc[c.doc_key] = per_doc.get(c.doc_key, 0) + 1
    print(f"       各文档块数：{per_doc}")

    if args.dry_run:
        print("[dry-run] 到此为止，未向量化、未入库。")
        for c in all_chunks[:3]:
            print("  ---", c.chunk_id, c.doc_key, f"p{c.page}", c.section[:40], "---")
            print("  ", c.text[:160].replace("\n", " "))
        imgc = next((c for c in all_chunks if c.type == "image"), None)
        if imgc:
            print("\n  [图像块示例]", imgc.chunk_id, imgc.section[:50])
            print("  ", imgc.text[:400].replace("\n", "\n   "))
        return 0

    print("\n[向量化] bge-small-zh-v1.5（CPU，本地）")
    from src.embedder import embed_texts

    t0 = time.perf_counter()
    vecs = embed_texts([c.text for c in all_chunks])
    print(f"      形状 {vecs.shape}，耗时 {time.perf_counter()-t0:.1f}s")

    print("[落盘]")
    from src.index_store import save_index

    out_dir = {"notable": config.INDEX_NOTABLE_DIR,
               "noimage": config.INDEX_NOIMAGE_DIR}.get(args.out)
    meta = save_index(all_chunks, vecs, source_paths, config.EMBEDDING_MODEL_PATH,
                      out_dir=out_dir)
    for k, v in meta.items():
        print(f"      {k}: {v}")
    print("\n[OK] 索引构建完成。启动服务：python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

