# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""从 PDF 抽取纯文本并落盘到 data/processed，用于逐页人工核对答案。

产物：
  data/processed/<stem>.txt       —— 带页码分隔符的全文本（供 grep 定位）
  data/processed/<stem>.pages.jsonl —— 每页一条 JSONL（page 从 1 开始）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pymupdf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "raw")
PROCESSED = os.path.join(REPO, "data", "processed")
PAGE_MARK = "\n<<<PAGE:{n}>>>\n"


def extract_one(path: str) -> tuple[str, list[dict]]:
    pages: list[dict] = []
    chunks: list[str] = []
    with pymupdf.open(path) as doc:
        for idx, page in enumerate(doc, start=1):
            try:
                text = page.get_text()
            except Exception as exc:  # 容错：单页解析失败不影响整体
                text = ""
                print(f"  [WARN] 第{idx}页解析失败: {exc!r}")
            pages.append({"page": idx, "text": text, "chars": len(text)})
            chunks.append(PAGE_MARK.format(n=idx))
            chunks.append(text)
    return "".join(chunks), pages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(PROCESSED, exist_ok=True)
    names = args.files or sorted(f for f in os.listdir(RAW) if f.lower().endswith(".pdf"))
    report = []
    for name in names:
        path = os.path.join(RAW, name)
        stem = os.path.splitext(name)[0]
        t0 = time.time()
        try:
            full, pages = extract_one(path)
        except Exception as exc:
            print(f"[ERR] {name} 打开失败: {exc!r}")
            continue
        out_txt = os.path.join(PROCESSED, stem + ".txt")
        with open(out_txt, "w", encoding="utf-8") as fh:
            fh.write(full)
        out_jsonl = os.path.join(PROCESSED, stem + ".pages.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as fh:
            for rec in pages:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        el = time.time() - t0
        empty = sum(1 for r in pages if r["chars"] < 20)
        report.append({"file": name, "pages": len(pages), "chars": sum(r["chars"] for r in pages),
                       "empty_pages": empty, "seconds": round(el, 2)})
        print(f"[OK] {name} 页={len(pages)} 字符={report[-1]['chars']} 空页={empty} 耗时={el:.2f}s")

    rp = os.path.join(PROCESSED, "_parse_report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("报告 ->", rp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
