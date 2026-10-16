# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：研发/data/eval/复核_probe_证据页扫描.py（T3 测试工程师，逐题证据页复核探针）

用途：逐题打开 PDF、抽取声明证据页正文，并统计每个判分关键字在**全文**中的命中页，
      为「证据页复核」提供机器可读证据（打印 + JSON 落盘）。
      —— 本脚本只读 PDF，不修改任何 golden 文件。

运行：
  pwsh -NoProfile -File run_py.ps1 研发/data/eval/复核_probe_证据页扫描.py
输出：
  研发/data/eval/复核_probe_证据页扫描.json
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "研发" / "data" / "raw"
PDFS = {
    "招股说明书1": RAW / "招股说明书1.pdf",
    "招股说明书2": RAW / "招股说明书2.pdf",
}

# 每题：声明证据页 + 判分关键字（用于确认「证据页确实含该证据」）
PROBES = [
    {"id": 1, "pdf": "招股说明书2", "pages": [22, 24],
     "keys": ["1,670", "1670", "25.04", "发行股数", "发行后总股本"]},
    {"id": 2, "pdf": "招股说明书2", "pages": [22, 30, 306, 144],
     "keys": ["3,393.40", "1,526.38", "2,492.78", "9,000.00",
              "其他与主营业务相关的营运资金", "仓储及物流中心", "研发中心",
              "电子商务平台", "扩充产品种类和数量"]},
    {"id": 3, "pdf": "招股说明书2", "pages": [157, 21, 158],
     "keys": ["赵马克", "42.35", "控股股东", "实际控制人", "存在控制关系的关联方"]},
    {"id": 4, "pdf": "招股说明书2", "pages": [157, 158],
     "keys": ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源",
              "力源贸易", "普芯达", "不存在控制关系的关联方"]},
    {"id": 5, "pdf": "招股说明书2", "pages": [39, 40, 113],
     "keys": ["销售部", "渠道销售部", "电话及网络销售部", "大客户销售部",
              "国际贸易部", "北京", "广州", "成都", "深圳", "武汉", "珠海",
              "组织结构图", "上海分公司"]},
    {"id": 6, "pdf": "招股说明书2", "pages": [72],
     "keys": ["IC", "增长率", "汽车", "工控", "计算机", "网络通信", "消费", "IC卡"]},
    {"id": 33, "pdf": "招股说明书1", "pages": [129],
     "keys": ["82.10", "97.31", "94.84", "94.34", "军用领域", "主营业务收入"]},
    {"id": 34, "pdf": "招股说明书1", "pages": [152],
     "keys": ["电子元器件制造企业", "金属壳体制造企业", "上游", "下游"]},
    {"id": 95, "pdf": "招股说明书1", "pages": [160],
     "keys": ["全军第一个视频指挥系统技术标准", "某视频指挥系统技术规范",
              "军事科学院", "重要供应商"]},
    {"id": 207, "pdf": "招股说明书1", "pages": [30, 479],
     "keys": ["15,000.00", "15,000", "补充流动资金", "募集资金"]},
    {"id": 260, "pdf": "招股说明书1", "pages": [129],
     "keys": ["6,464.51", "14,414.16", "18,780.67", "4,627.14", "军用领域的收入"]},
    {"id": 531, "pdf": "招股说明书1", "pages": [22],
     "keys": ["法定代表人", "程家明"]},
    {"id": 543, "pdf": "招股说明书1", "pages": [22],
     "keys": ["注册资本", "5,520"]},
    {"id": 793, "pdf": "招股说明书1", "pages": [152],
     "keys": ["下游", "军队", "政府机关", "能源"]},
    {"id": 795, "pdf": "招股说明书1", "pages": [27, 157],
     "keys": ["2014年12月", "某情报、指挥、控制与通信网络一体化工程",
              "国家科技进步一等奖", "C4ISR"]},
    {"id": 957, "pdf": "招股说明书1", "pages": [154, 332],
     "keys": ["重要供应商", "视频指挥", "国防军队"]},
]


def norm(s: str) -> str:
    """全角→半角、去空白、去千分位逗号、统一引号，用于稳健子串匹配。"""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)   # 千分位逗号
    s = re.sub(r"\s+", "", s)
    return s


def main() -> int:
    result = {"pdfs": {}, "probes": []}
    texts: dict[str, list[str]] = {}
    for name, path in PDFS.items():
        if not path.exists():
            print(f"[FATAL] 缺少 PDF: {path}")
            return 2
        with fitz.open(path) as doc:
            page_texts = [p.get_text("text") for p in doc]
            result["pdfs"][name] = {
                "path": str(path),
                "pages": doc.page_count,
                "sha256_head": None,
            }
        texts[name] = page_texts
        print(f"[OK] {name}: {len(page_texts)} 页  ({path})")

    for pr in PROBES:
        name = pr["pdf"]
        pages = pr["pages"]
        page_texts = texts[name]
        n = len(page_texts)
        rec = {
            "id": pr["id"], "pdf": name, "declared_pages": pages,
            "total_pages": n,
            "pages_in_range": all(1 <= p <= n for p in pages),
            "per_page": {},
            "keys_present_on_declared": {},
            "keys_page_index": {},
        }
        for p in pages:
            if not (1 <= p <= n):
                rec["per_page"][str(p)] = {"error": "out_of_range"}
                continue
            t = page_texts[p - 1]
            nt = norm(t)
            page_hits = {}
            for k in pr["keys"]:
                if norm(k) in nt:
                    page_hits[k] = True
            rec["per_page"][str(p)] = {
                "chars": len(t),
                "keys_hit": sorted(page_hits),
                "text_head": t[:600],
            }
        for k in pr["keys"]:
            nk = norm(k)
            pages_with = [i + 1 for i, t in enumerate(page_texts) if nk in norm(t)]
            rec["keys_page_index"][k] = pages_with[:40]
            rec["keys_present_on_declared"][k] = [
                p for p in pages if p in pages_with
            ]
        # 关键词整体覆盖度（关键字落在任一声明页 = 命中）
        rec["all_keys_covered"] = all(
            rec["keys_present_on_declared"][k] for k in pr["keys"]
        )
        rec["missing_keys"] = [
            k for k in pr["keys"] if not rec["keys_present_on_declared"][k]
        ]
        result["probes"].append(rec)
        print(f"\n===== 题 id={pr['id']}  [{name}]  声明证据页 {pages} =====")
        for k in pr["keys"]:
            on = rec["keys_present_on_declared"][k]
            pages_with = rec["keys_page_index"][k]
            print(f"  - {k!r:44s} 声明页命中={on}  全文命中页={pages_with[:12]}{'...' if len(pages_with) > 12 else ''}")

    out = ROOT / "研发" / "data" / "eval" / "复核_probe_证据页扫描.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[OK] 落盘：{out}")
    miss = [r["id"] for r in result["probes"] if not r["all_keys_covered"]]
    print(f"[汇总] 关键字未完全覆盖的题：{miss if miss else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
