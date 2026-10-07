"""
构建「优化前」朴素基线索引
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

用法：
    python scripts/build_baseline_index.py                     # 默认 data/raw 下的招股书
    python scripts/build_baseline_index.py --pdf data/raw/xxx.pdf
    python scripts/build_baseline_index.py --size 500 --overlap 50

耗时：548 页解析约 20~30s（不抽表格）+ 向量化约 1~2 分钟（本机 CPU）。
产物落盘在 data/index_naive/，与主线索引 data/index/ 完全隔离，互不影响。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.baseline import NaiveIndex, parse_naive  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(config.PDF_PATH))
    ap.add_argument("--size", type=int, default=config.NAIVE_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.NAIVE_CHUNK_OVERLAP)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"[ERR] PDF 不存在：{pdf}")
        return 1

    config.ensure_dirs()
    print(f"[工单] {config.WORK_ORDER_NO_OPT}")
    print(f"[基线] 朴素 RAG：定长 {args.size} 字滑窗 + 纯向量检索（无 BM25 / 无 DF 过滤 / 无闸门）")

    t0 = time.perf_counter()
    parsed = parse_naive(pdf)
    t_parse = time.perf_counter() - t0
    print(f"[解析] {parsed.total_pages} 页，{t_parse:.1f}s（未抽表格、未做行内标题断行）")

    t1 = time.perf_counter()
    idx = NaiveIndex.build(parsed, size=args.size, overlap=args.overlap)
    t_build = time.perf_counter() - t1

    lens = [len(c.text) for c in idx.chunks]
    print(f"[分块] {len(idx.chunks)} 块，平均 {sum(lens)/max(1,len(lens)):.0f} 字，"
          f"最长 {max(lens, default=0)} 字，向量化 {t_build:.1f}s")

    meta = idx.save({"source_file": pdf.name})
    print(f"[落盘] {config.INDEX_NAIVE_DIR}")
    for k in ("num_chunks", "embedding_dim"):
        print(f"  {k}: {meta[k]}")
    print(f"\n[OK] 总耗时 {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
