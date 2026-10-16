# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：研发/data/eval/复核_probe_逐页原文.py（T3，指定页逐字落盘）

用途：把每题关键证据页的 get_text("text") 原样落盘为
      研发/data/eval/复核_逐页原文/<pdf>_p<页码>.txt，供逐字比对（避免行聚类差异）。
只读 PDF。
运行：pwsh -NoProfile -File run_py.ps1 研发/data/eval/复核_probe_逐页原文.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "研发" / "data" / "raw"
OUT = ROOT / "研发" / "data" / "eval" / "复核_逐页原文"

WANT = {
    "招股说明书1": [22, 24, 26, 27, 53, 129, 152, 154, 157, 160, 330, 332, 479, 490],
    "招股说明书2": [21, 22, 24, 30, 38, 39, 40, 72, 113, 157, 158, 306, 144],
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in WANT.items():
        with fitz.open(RAW / f"{name}.pdf") as doc:
            for p in pages:
                t = doc[p - 1].get_text("text")
                f = OUT / f"{name}_p{p:03d}.txt"
                f.write_text(t, encoding="utf-8")
                print(f"[OK] {f}  ({len(t)} 字)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
