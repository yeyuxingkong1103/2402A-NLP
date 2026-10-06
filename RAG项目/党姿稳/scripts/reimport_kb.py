#!/usr/bin/env python
"""
reimport_kb.py — 把三个领域的 PDF 重新导入向量库

做两件事：

1. 补 page 字段。旧数据把页码编码在 source 里（"劳动法.pdf#p3"），
   现在页码是独立字段，回答末尾才拼得出"文件名 第N页"的引用。
2. 真正写进 Milvus。LOCAL_MODE=False 时数据落在 Milvus 集合里，
   打开 Milvus 页面能直接看到。

导入前会 drop 掉旧集合再重建：旧 schema 里没有 page / domain 字段，
而集合关掉了动态字段，不重建的话新字段会被静默丢弃（数据看着写进去了，实际没了）。

    python scripts/reimport_kb.py
    python scripts/reimport_kb.py --stats-only    # 只看统计，不动数据
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402
from vector_store import get_store  # noqa: E402

DOMAINS = ("legal", "medical", "english")

# data/generated_pdfs/ 下的文件名 → 领域。data/{domain}/ 里有 PDF 时优先用那些。
GENERATED_PDF_MAP = {
    "劳动法常见问题.pdf": "legal",
    "常见疾病诊疗指南.pdf": "medical",
    "中国高血压防治指南2024修订版.pdf": "medical",
    "英语语法精讲.pdf": "english",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重新导入三个领域的 PDF 到向量库")
    parser.add_argument("--stats-only", action="store_true", help="只打印各集合统计，不导入")
    return parser.parse_args()


def pdfs_for(domain: str) -> list[Path]:
    """该领域的 PDF：优先 data/{domain}/，为空则回退到 data/generated_pdfs/。"""
    own = sorted((config.DATA_DIR / domain).glob("*.pdf"))
    if own:
        return own
    return sorted(
        path
        for path in (config.DATA_DIR / "generated_pdfs").glob("*.pdf")
        if GENERATED_PDF_MAP.get(path.name) == domain
    )


def print_stats() -> None:
    """打印每个集合的条数、页码分布与来源文件，用来核对导入结果。"""
    store = get_store()
    total = 0

    for domain in DOMAINS:
        collection = config.KB_COLLECTIONS[domain]
        try:
            # Milvus 的 query 窗口上限是 offset+limit ≤ 16384，给大了会直接报错
            rows = store.query(collection, limit=16384)
        except Exception as exc:
            print(f"[{domain}] {collection}：读取失败（{exc}）")
            continue

        pages = sorted({int(row.get("page") or 0) for row in rows})
        sources = sorted({row.get("source") or "" for row in rows if row.get("source")})
        total += len(rows)
        print(f"[{domain}] {collection}：{len(rows)} 条")
        print(f"    页码：{pages}")
        print(f"    来源：{sources}")

    print(f"合计 {total} 条。")


def main() -> None:
    args = parse_args()

    mode = "本地文件" if config.LOCAL_MODE else f"Milvus({config.MILVUS_URI})"
    print(f"运行模式：{mode}　向量后端：{config.EMBEDDING_BACKEND}　维度：{config.EMBEDDING_DIM}")

    if args.stats_only:
        print_stats()
        return

    kb = KnowledgeBase()
    for domain in DOMAINS:
        pdfs = pdfs_for(domain)
        if not pdfs:
            print(f"[{domain}] 没找到 PDF，跳过。")
            continue

        kb.clear(domain)

        for pdf in pdfs:
            report = kb.import_pdf(pdf, domain)
            print(
                f"[{domain}] {report['file']}：{report['pages']} 页 → "
                f"原始 {report['raw_chunks']} 块 → 过滤后 {report['filtered_chunks']} 块 → "
                f"入库 {report['stored']} 条"
            )

    print()
    print_stats()


if __name__ == "__main__":
    main()
