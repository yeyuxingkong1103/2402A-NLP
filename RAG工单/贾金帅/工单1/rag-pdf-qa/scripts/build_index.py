"""
索引构建脚本
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
    python scripts/build_index.py                       # 用 .env 里的 PDF_PATH
    python scripts/build_index.py --pdf path/to/x.pdf   # 指定 PDF
    python scripts/build_index.py --no-tables           # 跳过表格抽取（快，约 1 分钟）
    python scripts/build_index.py --dry-run             # 只解析+分块，不向量化（看长度分布）

干跑（--dry-run）必须看两个数：平均块长 **和最长块长**。
只看平均会漏掉长尾巨块 —— 「平均正常 + 最长爆表」才是有问题的信号。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chunker import chunk_document  # noqa: E402
from src.pdf_parser import parse_pdf  # noqa: E402

WORK_ORDER_NO = config.WORK_ORDER_NO


def main() -> int:
    ap = argparse.ArgumentParser(description=f"构建索引 | 工单：{WORK_ORDER_NO}")
    ap.add_argument("--pdf", default=str(config.PDF_PATH), help="PDF 路径")
    ap.add_argument("--no-tables", action="store_true", help="跳过表格抽取")
    ap.add_argument("--dry-run", action="store_true", help="只解析与分块，不向量化不入库")
    args = ap.parse_args()

    config.ensure_dirs()
    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"[FAIL] PDF 不存在：{pdf}")
        return 2

    print(f"[工单] {WORK_ORDER_NO}")
    print(f"[1/4] 解析 PDF：{pdf.name}（表格抽取：{'关' if args.no_tables else '开'}）")
    t0 = time.perf_counter()
    parsed = parse_pdf(pdf, with_tables=not args.no_tables)
    print(f"      {parsed.total_pages} 页，耗时 {time.perf_counter()-t0:.1f}s")

    empty = sum(1 for p in parsed.pages if not p.text.strip())
    tbl_pages = sum(1 for p in parsed.pages if p.tables)
    print(f"      空文本页 {empty} 页；含表格页 {tbl_pages} 页")

    print("[2/4] 分块")
    t0 = time.perf_counter()
    chunks = chunk_document(parsed)
    lens = [len(c.text) for c in chunks] or [0]
    print(f"      {len(chunks)} 块，耗时 {time.perf_counter()-t0:.1f}s")
    print(
        f"      块长 平均 {sum(lens)/len(lens):.0f} / 最短 {min(lens)} / 最长 {max(lens)}"
        f"（上限 {config.CHUNK_MAX_LENGTH}）"
    )
    if max(lens) > config.CHUNK_MAX_LENGTH:
        print("[FAIL] 存在超上限的块，分块兜底失效！")
        return 3
    tables = [c for c in chunks if c.type == "table"]
    print(f"      其中表格块 {len(tables)} 条")

    if args.dry_run:
        print("[dry-run] 到此为止，未向量化、未入库。")
        for c in chunks[:3]:
            print("  ---", c.chunk_id, f"p{c.page}", c.section[:40], "---")
            print("  ", c.text[:160].replace("\n", " "))
        return 0

    print("[3/4] 向量化（bge-small-zh-v1.5，CPU）")
    from src.embedder import embed_texts

    config.ensure_dirs()
    t0 = time.perf_counter()
    vecs = embed_texts([c.text for c in chunks])
    print(f"      形状 {vecs.shape}，耗时 {time.perf_counter()-t0:.1f}s")

    print("[4/4] 落盘")
    from src.index_store import save_index

    meta = save_index(chunks, vecs, pdf, config.EMBEDDING_MODEL_PATH)
    for k, v in meta.items():
        print(f"      {k}: {v}")
    print("\n[OK] 索引构建完成。启动服务：python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
