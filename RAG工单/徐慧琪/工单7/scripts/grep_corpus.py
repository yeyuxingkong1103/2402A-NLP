# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
工单07 · 语料检索小工具（人工核对金标准用）

在 data/corpus/txt/ 的 9 份年报纯文本里做关键词检索，打印命中文件、字符偏移
与上下文。用途：为 10 道测试题**人工确认 gold 答案**（原值、年份、单位），
避免把 LLM 生成的内容当成金标准。

用法
----
    python scripts/grep_corpus.py "不良贷款率"              # 全语料检索
    python scripts/grep_corpus.py "拨备覆盖率" --doc 601658  # 限定某家公司
    python scripts/grep_corpus.py "逾期60天" --window 260 --max 5
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TXT_DIR = os.path.join(ROOT, "data", "corpus", "txt")


def iter_files(doc: str | None):
    if not os.path.isdir(TXT_DIR):
        raise SystemExit(f"语料目录不存在：{TXT_DIR}")
    for f in sorted(os.listdir(TXT_DIR)):
        if not f.endswith(".txt"):
            continue
        if doc and doc not in f:
            continue
        yield f, os.path.join(TXT_DIR, f)


def main() -> int:
    ap = argparse.ArgumentParser(description="语料关键词检索（人工核对金标准）")
    ap.add_argument("keyword", help="关键词或正则")
    ap.add_argument("--doc", default=None, help="限定文件名包含该串（如 601658）")
    ap.add_argument("--window", type=int, default=180, help="上下文前后字数")
    ap.add_argument("--max", type=int, default=6, help="每份文档最多打印命中数")
    ap.add_argument("--regex", action="store_true", help="把 keyword 当正则")
    ap.add_argument("--compact", action="store_true", help="只打印命中数与文件")
    args = ap.parse_args()

    pat = re.compile(args.keyword if args.regex else re.escape(args.keyword))
    total = 0
    for name, path in iter_files(args.doc):
        text = open(path, encoding="utf-8", errors="ignore").read()
        hits = list(pat.finditer(text))
        if not hits:
            continue
        total += len(hits)
        print(f"\n================ {name}  命中 {len(hits)} 处 ================")
        if args.compact:
            continue
        for m in hits[: args.max]:
            a = max(0, m.start() - args.window)
            b = min(len(text), m.end() + args.window)
            snippet = re.sub(r"\s+", " ", text[a:b])
            print(f"  @{m.start()}: ...{snippet}...")
    print(f"\n[合计] 命中 {total} 处")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
