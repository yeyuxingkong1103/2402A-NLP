# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""在已抽取的页文本中检索关键词，输出「页码 + 上下文」，用于逐页人工核对标准答案。"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(REPO, "data", "processed")

DOCS = {
    "1": "招股说明书1",       # 兴图新科
    "2": "招股说明书2",       # 力源信息
    "1w": "招股说明书1-无水印",
}


def load_pages(stem: str) -> list[dict]:
    path = os.path.join(PROCESSED, stem + ".pages.jsonl")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, choices=sorted(DOCS))
    ap.add_argument("--kw", nargs="+", required=True, help="关键词，多个则要求同一页内全部出现")
    ap.add_argument("--win", type=int, default=200, help="上下文窗口字符数")
    ap.add_argument("--max", type=int, default=12, help="最多输出多少条命中页")
    args = ap.parse_args()

    pages = load_pages(DOCS[args.doc])
    hits = 0
    for rec in pages:
        t = rec["text"]
        if all(k in t for k in args.kw):
            hits += 1
            if hits > args.max:
                continue
            print("=" * 78)
            print(f"[{DOCS[args.doc]}] 第 {rec['page']} 页")
            # 展示第一个关键词附近的上下文
            pos = t.find(args.kw[0])
            s = max(0, pos - args.win)
            e = min(len(t), pos + args.win * 3)
            print(re.sub(r"\n{2,}", "\n", t[s:e]).strip())
    print("-" * 78)
    print(f"命中页数 = {hits}（doc={args.doc}, kw={args.kw}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
