# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""探针：检查 PDF 图片尺寸分布与表格文本形态，用于校准过滤阈值。"""
from __future__ import annotations

import os
import sys
from collections import Counter

import pymupdf

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def main() -> int:
    for name in ["招股说明书1.pdf", "招股说明书2.pdf", "招股说明书1-无水印.pdf"]:
        with pymupdf.open(os.path.join(RAW, name)) as doc:
            sizes = Counter()
            big = 0
            for pno in range(min(doc.page_count, 60)):
                for info in doc[pno].get_images(full=True):
                    w, h = int(info[2]), int(info[3])
                    sizes[(w, h)] += 1
                    if w >= 300 and h >= 200:
                        big += 1
            print("=" * 70)
            print(name, "前60页图片尺寸 Top10:", sizes.most_common(10))
            print("   较大图(>=300x200) 数:", big)

    print("\n" + "=" * 70)
    print("招股说明书2 第22页 文本原始形态（看表格是否保留空白分隔）：")
    with pymupdf.open(os.path.join(RAW, "招股说明书2.pdf")) as doc:
        txt = doc[21].get_text()
        for line in txt.splitlines()[:35]:
            print(repr(line))
        print("\n--- find_tables() 结果 ---")
        try:
            tabs = doc[21].find_tables()
            print("表格数:", len(tabs.tables))
            for t in tabs.tables[:2]:
                print("  rows=", t.row_count, "cols=", t.col_count)
                print("  ", str(t.extract())[:400])
        except Exception as exc:
            print("find_tables 失败:", type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
