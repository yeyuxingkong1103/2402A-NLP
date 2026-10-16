# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""导出一页或一段页范围的全部文本（含空白折叠），用于人工逐页核对。"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(REPO, "data", "processed")

DOCS = {"1": "招股说明书1", "2": "招股说明书2", "1w": "招股说明书1-无水印"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, choices=sorted(DOCS))
    ap.add_argument("--from", dest="p0", type=int, required=True)
    ap.add_argument("--to", dest="p1", type=int, default=None)
    ap.add_argument("--raw", action="store_true", help="不折叠空白")
    args = ap.parse_args()
    p1 = args.p1 or args.p0
    path = os.path.join(PROCESSED, DOCS[args.doc] + ".pages.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.p0 <= rec["page"] <= p1:
                print(f"\n########## [{DOCS[args.doc]}] 第 {rec['page']} 页 ##########")
                text = rec["text"] if args.raw else "\n".join(
                    ln.rstrip() for ln in rec["text"].splitlines() if ln.strip()
                )
                print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
