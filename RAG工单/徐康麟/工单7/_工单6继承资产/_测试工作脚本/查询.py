# -*- coding: utf-8 -*-
"""工单7 辅助脚本：在页面文本缓存里做归一化检索，用于挑选「真实存在于 PDF」的关键词与数值。
用法: python 查询.py "关键词1" "关键词2" ... [--file 子串] [-n 3] [--ctx 160]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "pages_cache.json")

_FW = {ord(c): ord(c) - 0xFEE0 for c in
       "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
       "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ％（）．，－"}


def norm(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = s.translate(_FW)
    s = re.sub(r"[\s\u00a0]+", "", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s


def main() -> int:
    args = [a for a in sys.argv[1:]]
    ffilter = None
    nshow = 3
    ctx = 160
    terms = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--file":
            i += 1
            ffilter = args[i]
        elif a == "-n":
            i += 1
            nshow = int(args[i])
        elif a == "--ctx":
            i += 1
            ctx = int(args[i])
        else:
            terms.append(a)
        i += 1

    with open(CACHE, encoding="utf-8") as fh:
        cache = json.load(fh)

    for term in terms:
        nt = norm(term)
        print("=" * 20, term, "=" * 20)
        total = 0
        for name, pages in cache.items():
            if ffilter and ffilter not in name:
                continue
            hits = []
            for idx, raw in enumerate(pages):
                np_ = norm(raw)
                pos = np_.find(nt)
                if pos >= 0:
                    hits.append((idx + 1, np_[max(0, pos - ctx // 2): pos + ctx]))
            if hits:
                total += len(hits)
                print(f"[{name}] 命中 {len(hits)} 页: {[h[0] for h in hits][:nshow*4]}")
                for pg, snip in hits[:nshow]:
                    print(f"   p{pg}: ...{snip}...")
        if total == 0:
            print("   !! 未命中")
    return 0


if __name__ == "__main__":
    sys.exit(main())
