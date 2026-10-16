# -*- coding: utf-8 -*-
"""工单18 完善 A4：损坏文件归因分析。

工单编号：人工智能NLP-RAG-文档质量评估 Skill 工单

背景：全量报告里 `summary.corrupted_files = 83`（占 4.88%），但只记了「损坏」两个字，
无法行动。本脚本逐份复检这 83 份，归类到**可行动的失败原因**：

  * 零字节 / 极小文件
  * 非 PDF（扩展名伪装 / 内容非 %PDF-）
  * 截断（有 %PDF- 头但无 %%EOF，或 pymupdf 报 cannot open / damaged）
  * 加密（需要密码）
  * 页对象损坏（能打开但部分页读不出）
  * 其它（记录原始异常文本）

产出：``优化/评估结果/损坏文件归因.json`` + ``损坏文件清单.csv``（可直接拿去修复/剔除）

用法：& $py 研发/scripts/分析损坏文件.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import time

import pymupdf

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = pathlib.Path(r"E:\gao6gongdan\_共享输入\original_problems\documents")
REPORT = ROOT / "研发" / "data" / "reports" / "quality_report_full.json"
OUT_DIR = ROOT / "优化" / "评估结果"


def classify(path: pathlib.Path) -> dict:
    """对单份文件做归因，返回 {类别, 细节, 页数}。"""
    rec = {"文件": path.name, "字节": path.stat().st_size, "类别": "", "细节": "", "页数": None}
    size = rec["字节"]

    if size == 0:
        rec.update(类别="零字节", 细节="文件大小为 0")
        return rec

    head = path.open("rb").read(1024)
    if not head.startswith(b"%PDF-"):
        if head[:2] == b"PK":
            rec.update(类别="非PDF_实为ZIP类", 细节=f"头字节 {head[:4]!r}")
        elif head[:2] in (b"\xff\xd8", b"\x89P"):
            rec.update(类别="非PDF_实为图片", 细节=f"头字节 {head[:4]!r}")
        else:
            rec.update(类别="非PDF_其它", 细节=f"头字节 {head[:8]!r}")
        return rec

    # 有 PDF 头：看是否截断（缺 %%EOF）
    try:
        tail = path.open("rb").read()[-2048:]
        truncated = b"%%EOF" not in tail
    except Exception:
        truncated = False

    try:
        d = pymupdf.open(path)
        if d.needs_pass:
            rec.update(类别="加密需密码", 细节="needs_pass=True", 页数=d.page_count)
        else:
            n_bad = 0
            detail = ""
            for i in range(min(d.page_count, 20)):
                try:
                    _ = d[i].get_text()
                except Exception as e:
                    n_bad += 1
                    detail = f"第{i+1}页异常：{type(e).__name__}"
            if n_bad:
                rec.update(类别="页对象损坏", 细节=f"{n_bad}/20 页读取异常；{detail}", 页数=d.page_count)
            else:
                rec.update(类别="可打开_仅元数据异常",
                           细节="pymupdf 可正常打开；可能仅元数据/字体异常导致质检判损坏",
                           页数=d.page_count)
        d.close()
    except Exception as e:
        msg = str(e)
        if truncated:
            rec.update(类别="截断", 细节=f"{type(e).__name__}: {msg[:120]}（尾部无 %%EOF）")
        elif re.search(r"cannot open|damaged|broken|format error", msg, re.I):
            rec.update(类别="结构损坏", 细节=f"{type(e).__name__}: {msg[:120]}")
        else:
            rec.update(类别="其它", 细节=f"{type(e).__name__}: {msg[:120]}")
    return rec


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    pend = rep.get("pending_lists", {})
    names = [x if isinstance(x, str) else x.get("file") or x.get("name") or str(x)
             for x in (pend.get("corrupted_files") or [])]
    if not names:
        # 退化：从 errors 段取
        names = []
        for e in rep.get("errors") or []:
            n = e.get("file") if isinstance(e, dict) else None
            if n:
                names.append(n)
    names = sorted({pathlib.Path(n).name for n in names if n})
    print(f"报告中登记损坏文件 {len(names)} 份，开始逐份归因…")

    rows, t0 = [], time.perf_counter()
    for i, n in enumerate(names, 1):
        p = DOCS / n
        if not p.exists():
            rows.append({"文件": n, "字节": 0, "类别": "文件缺失", "细节": "报告中登记但磁盘不存在", "页数": None})
            continue
        try:
            rows.append(classify(p))
        except Exception as e:
            rows.append({"文件": n, "字节": p.stat().st_size, "类别": "归因失败",
                         "细节": f"{type(e).__name__}: {e}"[:120], "页数": None})
        if i % 20 == 0:
            print(f"  …{i}/{len(names)}")

    # 汇总
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["类别"]] = by_cat.get(r["类别"], 0) + 1
    total_bytes = sum(r["字节"] for r in rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "工单编号": "人工智能NLP-RAG-文档质量评估 Skill 工单",
        "完善项": "A4 损坏文件归因分析",
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "源报告": str(REPORT),
        "登记损坏数": len(rows),
        "占全量比例": round(len(rows) / rep["summary"]["total_files"], 4),
        "占用体积MB": round(total_bytes / 1024 / 1024, 2),
        "类别分布": dict(sorted(by_cat.items(), key=lambda x: -x[1])),
        "可行动结论": "",
        "逐份明细": rows,
        "耗时秒": round(time.perf_counter() - t0, 2),
    }
    # 给出可行动结论
    act = []
    if by_cat.get("零字节"):
        act.append(f"零字节 {by_cat['零字节']} 份 → 直接删除或重新下载")
    if by_cat.get("截断"):
        act.append(f"截断 {by_cat['截断']} 份 → 原文件传输不完整，需重新获取")
    if by_cat.get("加密需密码"):
        act.append(f"加密 {by_cat['加密需密码']} 份 → 需密码或换源")
    nonpdf = sum(v for k, v in by_cat.items() if k.startswith("非PDF"))
    if nonpdf:
        act.append(f"非 PDF {nonpdf} 份 → 扩展名伪装，应从 PDF 语料中剔除")
    if by_cat.get("页对象损坏"):
        act.append(f"页对象损坏 {by_cat['页对象损坏']} 份 → 可尝试 pymupdf 修复后重新解析")
    if by_cat.get("可打开_仅元数据异常"):
        act.append(f"可打开 {by_cat['可打开_仅元数据异常']} 份 → **严重度被高估**，"
                   f"实际可解析，建议复核质检判损坏的阈值")
    payload["可行动结论"] = act

    (OUT_DIR / "损坏文件归因.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "损坏文件清单.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["文件", "字节", "类别", "细节", "页数"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] {OUT_DIR / '损坏文件归因.json'}")
    print(f"[OK] {OUT_DIR / '损坏文件清单.csv'}（{len(rows)} 行）")
    print("  类别分布：")
    for k, v in payload["类别分布"].items():
        print(f"    {k:22} {v:3} 份")
    print("  可行动结论：")
    for a in act:
        print(f"    · {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
