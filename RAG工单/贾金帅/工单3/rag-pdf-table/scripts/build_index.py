"""
索引构建脚本（工单03：多文档 + 表格解析）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用法：
    python scripts/build_index.py                      # 按 config.DOCS 建两份文档的索引
    python scripts/build_index.py --no-tables          # 关掉表格解析（**工单03 的对照组**）
    python scripts/build_index.py --dry-run            # 只解析 + 分块，不向量化不入库
    python scripts/build_index.py --only liyuan        # 只建其中一份（调试用）

--no-tables 就是工单03 要对比的「优化前」：表格内容不做结构化，
而是作为普通文本行混进段落参与定长/标题感知分块（工单1/2 的做法）。
两份索引的检索算法完全一致，只有「表格有没有被结构化」这一个变量不同。

干跑（--dry-run）必须看两个数：平均块长 **和最长块长**。
只看平均会漏掉长尾巨块 —— 「平均正常 + 最长爆表」才是有问题的信号。
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chunker import chunk_document  # noqa: E402
from src.pdf_parser import parse_pdf  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO_TABLE


def resolve_docs(only: str) -> list[dict]:
    docs = config.DOCS
    if only:
        docs = [d for d in docs if d["key"] == only]
        if not docs:
            raise SystemExit(f"[FAIL] config.DOCS 里没有 key={only}")
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=f"构建索引 | 工单：{WORK_ORDER_NO}")
    ap.add_argument("--no-tables", action="store_true",
                    help="跳过表格解析（工单03 的「优化前」对照组）")
    ap.add_argument("--dry-run", action="store_true", help="只解析与分块，不向量化不入库")
    ap.add_argument("--only", default="", help="只建指定 key 的文档（xingtu / liyuan）")
    ap.add_argument("--out", default="", choices=["", "notable"],
                    help="notable = 落到 data/index_notable/（工单03 的对照组索引目录）")
    args = ap.parse_args()

    config.ensure_dirs()
    docs = resolve_docs(args.only)
    with_tables = not args.no_tables

    print(f"[工单] {WORK_ORDER_NO}")
    print(f"[语料] {len(docs)} 份文档，表格解析：{'开' if with_tables else '关（对照组）'}\n")

    all_chunks = []
    all_tables = 0
    source_paths: list[Path] = []
    seq = 0

    for di, d in enumerate(docs, 1):
        pdf = config.RAW_DIR / d["file"]
        if not pdf.exists():
            print(f"[FAIL] 文档不存在：{pdf}")
            return 2
        source_paths.append(pdf)

        print(f"[{di}/{len(docs)}] 解析 {pdf.name} —— {d['name']}")
        t0 = time.perf_counter()
        parsed = parse_pdf(
            pdf,
            with_tables=with_tables,
            header_res=[re.compile(d["header_re"])],   # 页眉按文档给，避免漏剔另一家的页眉
            doc_name=d["name"],
        )
        parse_s = time.perf_counter() - t0
        empty = sum(1 for p in parsed.pages if not p.text.strip())
        tbl_pages = len({t.page for t in parsed.tables})
        n_tables = len(parsed.tables)
        all_tables += n_tables
        print(f"      {parsed.total_pages} 页 / 耗时 {parse_s:.1f}s；"
              f"空文本页 {empty}；结构化表格 {n_tables} 张（分布在 {tbl_pages} 页）")

        t0 = time.perf_counter()
        chunks = chunk_document(parsed, doc_key=d["key"], seq_start=seq)
        seq += len(chunks)
        lens = [len(c.text) for c in chunks] or [0]
        n_tbl_chunks = sum(1 for c in chunks if c.type == "table")
        print(f"      分块 {len(chunks)} 条（表格块 {n_tbl_chunks}），"
              f"块长 平均 {sum(lens)/len(lens):.0f} / 最短 {min(lens)} / 最长 {max(lens)}"
              f"（上限 {config.CHUNK_MAX_LENGTH}），耗时 {time.perf_counter()-t0:.1f}s")
        if max(lens) > config.CHUNK_MAX_LENGTH:
            print("[FAIL] 存在超上限的块，分块兜底失效！")
            return 3
        all_chunks.extend(chunks)

    lens = [len(c.text) for c in all_chunks] or [0]
    print(f"\n[合计] {len(all_chunks)} 块（表格块 {sum(1 for c in all_chunks if c.type=='table')}），"
          f"表格 {all_tables} 张")
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
        return 0

    print("\n[向量化] bge-small-zh-v1.5（CPU，本地）")
    from src.embedder import embed_texts

    t0 = time.perf_counter()
    vecs = embed_texts([c.text for c in all_chunks])
    print(f"      形状 {vecs.shape}，耗时 {time.perf_counter()-t0:.1f}s")

    print("[落盘]")
    from src.index_store import save_index

    meta = save_index(all_chunks, vecs, source_paths, config.EMBEDDING_MODEL_PATH,
                      out_dir=config.INDEX_NOTABLE_DIR if args.out == "notable" else None)
    for k, v in meta.items():
        print(f"      {k}: {v}")
    print("\n[OK] 索引构建完成。启动服务：python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
