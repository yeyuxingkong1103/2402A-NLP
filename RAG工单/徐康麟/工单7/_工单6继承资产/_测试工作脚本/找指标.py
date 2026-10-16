# -*- coding: utf-8 -*-
"""工单7 辅助脚本：为每份年报定位「主要会计数据/财务摘要」正文页（跳过目录页）并打印。
用法: python 找指标.py [文件子串] [--chars 2500]
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


def dots_ratio(raw: str) -> float:
    if not raw:
        return 1.0
    return raw.count(".") / max(1, len(raw))


def main() -> int:
    args = sys.argv[1:]
    chars = 2500
    key = None
    if "--chars" in args:
        i = args.index("--chars")
        chars = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if args:
        key = args[0]
    cache = json.load(open(CACHE, encoding="utf-8"))
    for name, pages in cache.items():
        if key and key not in name:
            continue
        for idx, raw in enumerate(pages):
            n = norm(raw)
            if dots_ratio(raw) > 0.03:
                continue
            if ("营业收入" in n and ("归属于" in n or "净利润" in n)) and \
               ("本期" in n or "本年" in n or "同比" in n or "增减" in n):
                print("=" * 20, name, "PAGE", idx + 1, "=" * 20)
                print(raw[:chars])
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
