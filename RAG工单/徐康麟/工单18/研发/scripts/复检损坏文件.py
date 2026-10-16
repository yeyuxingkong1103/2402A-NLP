# -*- coding: utf-8 -*-
"""工单18 完善 A4（简化版）：复检「损坏文件」的真实性质。

工单编号：人工智能NLP-RAG-文档质量评估 Skill 工单

发现：全量报告里 83 份的 error 文本是 ``unhandled: MemoryError:``，
**不是文件损坏**，而是处理端内存不足。本脚本抽样复检若干份，验证该判断，
给出「质检结论修正」需要的证据。产出 ``优化/评估结果/损坏文件复检.json``。

用法：& $py 研发/scripts/复检损坏文件.py [--sample 8]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import pymupdf

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT = ROOT / "研发" / "data" / "reports" / "quality_report_full.json"
OUT = ROOT / "优化" / "评估结果" / "损坏文件复检.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=8, help="抽检份数（均匀抽样）")
    args = ap.parse_args()

    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    items = (rep.get("pending_lists") or {}).get("corrupted_files") or []
    err_kinds: dict[str, int] = {}
    for it in items:
        e = str(it.get("error", ""))
        key = e.split(":")[0].strip() or "未知"
        err_kinds[key] = err_kinds.get(key, 0) + 1

    step = max(1, len(items) // args.sample)
    picked = items[::step][:args.sample]

    rows = []
    for it in picked:
        p = pathlib.Path(it["file"])
        rec = {"文件": p.name, "报告记录大小": it.get("size_bytes"),
               "报告错误": it.get("error"), "磁盘存在": p.exists()}
        if p.exists():
            rec["磁盘大小"] = p.stat().st_size
            rec["大小一致"] = rec["磁盘大小"] == it.get("size_bytes")
            try:
                d = pymupdf.open(p)
                rec["可打开"] = True
                rec["页数"] = d.page_count
                n = 0
                chars = 0
                for i in range(min(d.page_count, 5)):
                    t = d[i].get_text().strip()
                    chars += len(t)
                    if len(t) > 20:
                        n += 1
                rec["前5页有文本的页数"] = n
                rec["前5页文本字符数"] = chars
                d.close()
            except Exception as e:
                rec["可打开"] = False
                rec["异常"] = f"{type(e).__name__}: {str(e)[:100]}"
        rows.append(rec)

    ok = sum(1 for r in rows if r.get("可打开"))
    payload = {
        "工单编号": "人工智能NLP-RAG-文档质量评估 Skill 工单",
        "完善项": "A4 复检「损坏文件」真实性质",
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "报告登记数": len(items),
        "报告错误类型分布": dict(sorted(err_kinds.items(), key=lambda x: -x[1])),
        "抽检份数": len(rows),
        "抽检可打开数": ok,
        "结论": "",
        "明细": rows,
    }
    if ok == len(rows) and rows:
        payload["结论"] = (
            f"抽检 {len(rows)} 份**全部可正常打开**，且报告错误文本为 `MemoryError`，"
            f"因此这 {len(items)} 份**并非文件损坏**，而是质检处理端内存不足导致的漏检。"
            "建议：① 把质检结论从「损坏」改判为「未分析」；② 用更省内存的解析参数重跑这些文件"
            "（如限制单页超时、分页解析、降低并发）。"
        )
    else:
        payload["结论"] = f"抽检 {len(rows)} 份中 {ok} 份可打开，需进一步逐份判定。"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {OUT}")
    print(f"  报告登记 {len(items)} 份，错误类型分布：{payload['报告错误类型分布']}")
    print(f"  抽检 {len(rows)} 份，可正常打开 {ok} 份")
    print(f"  结论：{payload['结论'][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
