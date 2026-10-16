# -*- coding: utf-8 -*-
"""工单7 辅助脚本：定位「主要会计数据和财务指标」页并整页打印，用于抄录真实数值。
用法: python 看页.py 文件子串 [关键词] [页数上限]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "pages_cache.json")


def norm(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"[\s\u00a0]+", "", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s


def main() -> int:
    key = sys.argv[1]
    kw = sys.argv[2] if len(sys.argv) > 2 else "主要会计数据和财务指标"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    cache = json.load(open(CACHE, encoding="utf-8"))
    shown = 0
    for name, pages in cache.items():
        if key not in name:
            continue
        for idx, raw in enumerate(pages):
            if norm(kw) in norm(raw):
                print("=" * 25, name, "PAGE", idx + 1, "=" * 25)
                print(raw)
                shown += 1
                if shown >= limit:
                    return 0
    if shown == 0:
        print("!! 未找到", key, kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
