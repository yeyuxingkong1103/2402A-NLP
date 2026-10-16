# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""临时探针：确认两份招股说明书各自的发行人与规模。"""
from __future__ import annotations

import os
import sys

import pymupdf

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
PATS = ["武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司"]


def main() -> int:
    for name in ["招股说明书1.pdf", "招股说明书2.pdf", "招股说明书1-无水印.pdf"]:
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print("缺失", name)
            continue
        with pymupdf.open(path) as doc:
            full = "".join(page.get_text() for page in doc)
            print("=" * 70)
            print(name, "pages=", doc.page_count, "chars=", len(full))
            for pat in PATS:
                print("   ", pat, full.count(pat))
            print("    head:", full[:100].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
