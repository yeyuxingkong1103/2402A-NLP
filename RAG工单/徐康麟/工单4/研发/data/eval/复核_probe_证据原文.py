# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：研发/data/eval/复核_probe_证据原文.py（T3，逐题证据原文/页脚页码复核探针）

用途：对每题声明证据页，抽「证据原文窗口」与「页脚（y 最大的 span）」，
      输出 研发/data/eval/复核_probe_证据原文.json，供人工逐题比对 golden 的 answer/evidence。
      只读 PDF。

运行：pwsh -NoProfile -File run_py.ps1 研发/data/eval/复核_probe_证据原文.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "研发" / "data" / "raw"

TARGETS = [
    # (id, pdf, page, 锚点正则)
    (1, "招股说明书2", 22, r"发行股数|发行后总股本|发行股份数量"),
    (1, "招股说明书2", 24, r"发行股数|发行后总股本|1,670"),
    (2, "招股说明书2", 22, r"募集资金拟投资项目|仓储及物流中心|其他与主营业务相关"),
    (2, "招股说明书2", 30, r"其他与主营业务相关|仓储及物流中心|募集资金"),
    (2, "招股说明书2", 306, r"3,393.40|募集资金投资项目的具体情况"),
    (2, "招股说明书2", 144, r"仓储及物流中心|电子商务平台"),
    (3, "招股说明书2", 157, r"存在控制关系的关联方"),
    (3, "招股说明书2", 21, r"42.35"),
    (3, "招股说明书2", 158, r"控股股东|实际控制人"),
    (4, "招股说明书2", 157, r"不存在控制关系的关联方"),
    (4, "招股说明书2", 158, r"普芯达"),
    (5, "招股说明书2", 39, r"销售部|国际贸易部|上海分公司"),
    (5, "招股说明书2", 40, r"渠道销售部|大客户销售部"),
    (5, "招股说明书2", 113, r"大客户销售部|销售处"),
    (6, "招股说明书2", 72, r"增长率|中国IC市场|应用结构"),
    (33, "招股说明书1", 129, r"占主营业务收入比重"),
    (260, "招股说明书1", 129, r"军用领域的收入"),
    (34, "招股说明书1", 152, r"行业上下游情况"),
    (793, "招股说明书1", 152, r"下游行业为各类终端用户"),
    (95, "招股说明书1", 160, r"全军第一个视频指挥系统技术标准"),
    (207, "招股说明书1", 30, r"补充流动资金"),
    (207, "招股说明书1", 479, r"补充流动资金"),
    (531, "招股说明书1", 22, r"法定代表人"),
    (543, "招股说明书1", 22, r"注册资本"),
    (795, "招股说明书1", 27, r"一体化工程"),
    (795, "招股说明书1", 157, r"一体化工程"),
    (957, "招股说明书1", 154, r"重要供应商"),
    (957, "招股说明书1", 332, r"重要供应商|视频指挥"),
]


def footer(page: fitz.Page) -> dict:
    """取页脚：y 最大的 span（T1 结论：印刷页码 = 物理页码 − 1，页脚须取 y 最大 span）。"""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                spans.append(sp)
    if not spans:
        return {"y_max": None, "text": None}
    sp = max(spans, key=lambda s: s["bbox"][3])
    return {"y_max": round(sp["bbox"][3], 1), "x0": round(sp["bbox"][0], 1),
            "text": sp["text"].strip()}


def main() -> int:
    out = []
    for qid, pdfname, page_no, pattern in TARGETS:
        path = RAW / f"{pdfname}.pdf"
        with fitz.open(path) as doc:
            page = doc[page_no - 1]
            text = page.get_text("text")
            n_tables = len(page.find_tables().tables)
            ft = footer(page)
        rx = re.compile(pattern)
        windows = []
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            if rx.search(ln):
                lo, hi = max(0, i - 4), min(len(lines), i + 12)
                windows.append({"line": i, "window": "\n".join(lines[lo:hi])})
        out.append({
            "id": qid, "pdf": pdfname, "page": page_no, "pattern": pattern,
            "n_tables": n_tables, "footer": ft,
            "printed_page_guess": ft["text"],
            "chars": len(text),
            "windows": windows[:4],
        })
        print(f"\n############ id={qid} {pdfname} 物理页 {page_no}  表格数={n_tables} 页脚={ft['text']!r} ############")
        for w in windows[:3]:
            print(w["window"])
            print("  --- (窗口分割) ---")
    p = ROOT / "研发" / "data" / "eval" / "复核_probe_证据原文.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[OK] 落盘：{p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
