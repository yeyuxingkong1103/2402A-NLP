# -*- coding: utf-8 -*-
"""工单18 完善：对扫描型 PDF 抽样调用百度 OCR，评估「补 OCR」的收益与额度代价。

工单编号：人工智能NLP-RAG-文档质量评估 Skill 工单

背景：全量报告修正后，1700 份中 **Scan_PDF 622 份（36.6%）** 无文本层，
      原报告只能记为「扫描型 → 需 OCR」，而本机此前无 OCR 引擎。
      接入百度 OCR 后，本脚本抽样验证：**补 OCR 能拿到多少内容、耗时与额度代价多大**。

产出：``优化/评估结果/扫描件OCR抽样.json``
用法：& $py 研发/scripts/扫描件OCR抽样.py --sample 10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import pymupdf

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, r"E:\gao6gongdan\工单14\研发\scripts")
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "bdocr", pathlib.Path(r"E:\gao6gongdan\工单14\研发\scripts\百度OCR.py"))
bdocr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdocr)  # type: ignore[union-attr]

REPORT = ROOT / "研发" / "data" / "reports" / "quality_report_full.json"
DOCS = pathlib.Path(r"E:\gao6gongdan\_共享输入\original_problems\documents")
OUT = ROOT / "优化" / "评估结果" / "扫描件OCR抽样.json"  # 可被 --out 覆盖
TMP = ROOT / "优化" / "评估结果" / "_ocr_tmp"


def scan_pdfs() -> list[str]:
    """从全量报告的 per_file 段取扫描型清单（字段名实测为 pdf_type / name）。"""
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    pf = rep.get("per_file") or []
    names = [str(r.get("name") or r.get("path") or "")
             for r in pf if str(r.get("pdf_type", "")).upper() == "SCAN_PDF"]
    if not names:
        names = [str(x.get("file") or "")
                 for x in ((rep.get("pending_lists") or {}).get("pdf_type_pending_confirm") or [])
                 if str(x.get("pdf_type", "")).upper() == "SCAN_PDF"]
    return [pathlib.Path(n).name for n in names if n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10, help="抽样份数（均匀抽样）")
    ap.add_argument("--offset", type=int, default=0, help="从均匀抽样序列的第 N 份开始（避免与既有抽样重叠）")
    ap.add_argument("--out", default=None, help="结果 JSON 路径（默认 优化/评估结果/扫描件OCR抽样.json）")
    ap.add_argument("--max-pages", type=int, default=3, help="每份最多 OCR 前 N 页（控额度）")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    scans = scan_pdfs()
    if not scans:
        print("[!!] 未能从报告里取到扫描型文件清单")
        return 3
    step = max(1, len(scans) // (args.sample + args.offset)) if (args.sample + args.offset) else 1
    seq = scans[::step]
    picked = seq[args.offset: args.offset + args.sample]
    if not picked:
        picked = scans[:args.sample]

    cred = json.loads((ROOT.parent / "工单14" / "研发" / "data" / "百度OCR凭据.json")
                      .read_text(encoding="utf-8"))
    token = bdocr.get_token(cred["api_key"], cred["secret_key"])
    TMP.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.perf_counter()
    calls = 0
    for name in picked:
        pdf = DOCS / name
        rec = {"文件": name, "存在": pdf.exists()}
        if not pdf.exists():
            rows.append(rec)
            continue
        rec["字节"] = pdf.stat().st_size
        try:
            d = pymupdf.open(pdf)
            rec["页数"] = d.page_count
            n_pages = min(args.max_pages, d.page_count)
            chars = 0
            lines = 0
            ok_pages = 0
            for i in range(n_pages):
                img = TMP / f"{pdf.stem}_p{i+1}.png"
                d[i].get_pixmap(dpi=args.dpi).save(img)
                resp = bdocr.ocr_image(token, img, retries=2)
                calls += 1
                if "words_result" in resp:
                    ok_pages += 1
                    lines += len(resp["words_result"])
                    chars += len(bdocr.extract_text(resp))
                else:
                    rec.setdefault("错误", []).append(f"p{i+1}: {resp.get('error_code')} {resp.get('error_msg')}")
                img.unlink(missing_ok=True)
            d.close()
            rec.update({"OCR页数": n_pages, "成功页": ok_pages,
                        "识别行数": lines, "识别字符数": chars})
        except Exception as e:
            rec["异常"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
        print(f"  {name}: {rec.get('成功页','-')}/{rec.get('OCR页数','-')} 页, "
              f"{rec.get('识别字符数',0)} 字符")

    tot_pages = sum(r.get("OCR页数") or 0 for r in rows)
    tot_chars = sum(r.get("识别字符数") or 0 for r in rows)
    ok_files = sum(1 for r in rows if (r.get("成功页") or 0) > 0)
    payload = {
        "工单编号": "人工智能NLP-RAG-文档质量评估 Skill 工单",
        "完善项": "扫描型 PDF 补 OCR 抽样评估",
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "扫描型总数": len(scans),
        "抽样份数": len(rows),
        "每份最多页数": args.max_pages,
        "渲染DPI": args.dpi,
        "OCR调用次数": calls,
        "成功文件数": ok_files,
        "OCR页数合计": tot_pages,
        "识别字符合计": tot_chars,
        "平均每页字符": round(tot_chars / tot_pages, 1) if tot_pages else 0,
        "耗时秒": round(time.perf_counter() - t0, 1),
        "额度估算": {
            "本次消耗调用": calls,
            "若全量补OCR（622 份 × 平均页数）": "见下方推算",
            "推算": "",
        },
        "逐份": rows,
    }
    avg_pages = (sum(r.get("页数") or 0 for r in rows) / len(rows)) if rows else 0
    est_calls = int(len(scans) * avg_pages)
    payload["额度估算"]["推算"] = (
        f"抽样平均页数 {avg_pages:.1f} → 全量 {len(scans)} 份约需 **{est_calls} 次 OCR 调用**"
        f"（本次抽样 {calls} 次，耗时 {payload['耗时秒']}s → 全量约 {est_calls * payload['耗时秒'] / max(1, calls) / 60:.0f} 分钟）"
    )
    out_path = pathlib.Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT = out_path
    try:
        TMP.rmdir()
    except OSError:
        pass

    print(f"\n[OK] {OUT}")
    print(f"  抽样 {len(rows)} 份 / OCR {calls} 次调用 / {tot_pages} 页 / {tot_chars} 字符"
          f"（平均 {payload['平均每页字符']} 字符/页）")
    print(f"  全量推算：{payload['额度估算']['推算']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
