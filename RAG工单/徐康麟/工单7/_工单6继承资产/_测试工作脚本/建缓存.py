# -*- coding: utf-8 -*-
"""工单7 辅助脚本：把语料 9 份年报逐页文本抽成 JSON 缓存，供关键词核对/探查快速复用。
不联网、不调用任何模型、不写语料目录以外的位置。
"""
import json
import os
import sys

import pymupdf

RAW = r"E:\gao6gongdan\工单7\研发\data\raw"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages_cache.json")


def main() -> int:
    files = sorted(f for f in os.listdir(RAW) if f.lower().endswith(".pdf"))
    cache = {}
    for name in files:
        path = os.path.join(RAW, name)
        doc = pymupdf.open(path)
        pages = []
        for pg in doc:
            pages.append(pg.get_text("text"))
        cache[name] = pages
        print(f"{name}\tpages={len(pages)}")
        doc.close()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    print("CACHE_OK", OUT, os.path.getsize(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
