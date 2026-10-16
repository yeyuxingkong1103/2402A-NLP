# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：研发/data/eval/复核_probe_跨文件混淆项.py（T3，双 PDF 交叉混淆项核对）

用途：找出「两份 PDF 里同字段不同值」的事实，写进 golden 的禁止误答项/混淆项。
只读 PDF。
运行：pwsh -NoProfile -File run_py.ps1 研发/data/eval/复核_probe_跨文件混淆项.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "研发" / "data" / "raw"

FIELDS = ["注册资本", "法定代表人", "发行股数", "控股股东", "实际控制人", "补充流动资金"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return re.sub(r"\s+", "", s)


def main() -> int:
    for name in ["招股说明书1", "招股说明书2"]:
        print(f"\n########## {name} ##########")
        with fitz.open(RAW / f"{name}.pdf") as doc:
            for i, page in enumerate(doc):
                t = norm(page.get_text("text"))
                for f in FIELDS:
                    for m in re.finditer(re.escape(f), t):
                        seg = t[m.end():m.end() + 30]
                        if re.match(r"^[:：]?[\u4e00-\u9fa5A-Za-z0-9,.]", seg) and (
                                any(ch.isdigit() for ch in seg[:12]) or f in ("法定代表人", "控股股东", "实际控制人")):
                            print(f"  p{i + 1:>3} {f} -> {seg[:34]}")
                            break
                if i > 40:
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
