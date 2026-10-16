# -*- coding: utf-8 -*-
"""工单18 测试截图生成（中文命名，数据全部来自 1700 份实跑报告）。

工单编号：人工智能NLP-RAG-文档质量评估 Skill 工单
数据源：研发/data/reports/quality_report_full.json
产出（测试/测试截图/）：
  整体评估汇总_1700份文档.png
  PDF类型分流_文本与扫描占比.png
  文档长度分布_按类型.png
  敏感信息检测与待审清单.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, r"E:\gao6gongdan\_共享工具")
from 截图渲染 import Page, load_json  # noqa: E402

REPORT = ROOT / "研发" / "data" / "reports" / "quality_report_full.json"
SHOT = ROOT / "测试" / "测试截图"
SHOT.mkdir(parents=True, exist_ok=True)
SRC = "研发/data/reports/quality_report_full.json"


def main() -> int:
    d = load_json(REPORT)
    if not d:
        print(f"[!!] 缺报告：{REPORT}")
        return 3
    s = d["summary"]
    run = d["run"]
    fmt = d["format_distribution"]
    pdf = d["pdf_type_analysis"]
    ln = d["length_distribution"]
    dup = d["duplicate_detection"]
    sens = d["sensitive_detection"]
    tags = d["tag_system"]
    pend = d["pending_lists"]

    # ------------------------------------------------ 图1 整体汇总
    p = Page("工单18 整体评估汇总（1700 份工业文档实测）",
             f"数据源 quality_report_full.json｜分析耗时 {run['wall_clock_sec']:.1f} s")
    p.section("1. 规模与健康度")
    p.kv_block([
        ("文件总数", f"{s['total_files']:,}"),
        ("总大小", f"{s['total_mb']:.1f} MB"),
        ("损坏文件", f"{s['corrupted_files']}（{s['corrupted_files'] / s['total_files'] * 100:.1f}%）"),
        ("MD5 完全重复组", str(s["md5_duplicate_groups"])),
        ("版本冲突（近重复）", str(len(dup["simhash_near"]))),
        ("敏感信息命中", str(sens["total_hits"])),
    ], cols=2)
    p.section("2. 格式分布（按扩展名 / 按组）")
    ext = (fmt.get("by_extension") or [])[:6]
    grp = (fmt.get("by_group") or [])[:6]
    rows_fmt = [[x["name"], f"{x['count']:,}", f"{x['ratio'] * 100:.2f}%",
                 f"{x['bytes'] / 1024 / 1024:.1f} MB"] for x in ext + grp]
    p.table_block(["名称", "数量", "占比", "体积"], rows_fmt, widths=[2, 2, 2, 2])
    p.section("3. 分类标签分布（主导标签）")
    dist = (tags.get("distribution") or {}).get("by_primary") or []
    p.table_block(["标签", "数量", "占比"],
                  [[x["name"], f"{x['count']:,}", f"{x['ratio'] * 100:.2f}%"] for x in dist[:8]],
                  widths=[3, 2, 2])
    p.finish(SHOT / "整体评估汇总_1700份文档.png", sources=[SRC])

    # ------------------------------------------------ 图2 PDF 分流
    p2 = Page("工单18 PDF 类型分流：文本型 vs 扫描型",
              "决定解析路由的关键指标：扫描型必须走 OCR 路径")
    p2.section("1. 分流结果")
    counts = pdf["counts"]
    ratios = pdf["ratios"]
    p2.table_block(["类型", "数量", "占全部 PDF 比例", "解析路由建议"], [
        ["Text_PDF（文本型）", str(counts.get("Text_PDF", 0)),
         f"{ratios.get('Text_PDF', 0) * 100:.2f}%", "DeepDOC 文本路径"],
        ["Scan_PDF（扫描型）", str(counts.get("Scan_PDF", 0)),
         f"{ratios.get('Scan_PDF', 0) * 100:.2f}%", "OCR 路径（本机无 OCR 引擎）"],
        ["Mixed_PDF（混合型）", str(counts.get("Mixed_PDF", 0)),
         f"{ratios.get('Mixed_PDF', 0) * 100:.2f}%", "逐页判定后分流"],
        ["Corrupted（损坏）", str(counts.get("Corrupted", 0)),
         f"{ratios.get('Corrupted', 0) * 100:.2f}%", "隔离并记录"],
    ], widths=[4, 2, 3, 5])
    p2.section("2. 页级抽样证据")
    p2.kv_block([
        ("PDF 文件数", str(pdf["pdf_file_count"])),
        ("解析成功", str(pdf["parsed_ok"])),
        ("抽样总页数", str(pdf["sampled_pages_total"])),
        ("扫描型页数", str(pdf["scanned_pages_total"])),
        ("低文本页数", str(pdf["low_text_pages_total"])),
        ("扫描型占比", f"{s['scan_pdf_ratio'] * 100:.2f}%"),
    ], cols=2)
    p2.note("判定口径：抽样页面文本层字符数低于阈值即判为扫描型；阈值配置在 研发/assessment_config.yaml")
    p2.finish(SHOT / "PDF类型分流_文本与扫描占比.png", sources=[SRC])

    # ------------------------------------------------ 图3 长度分布
    p3 = Page("工单18 文档长度分布（按类型）", "长度分布用于容量规划与分块策略选择")
    p3.section("1. 全体与分层（字符数分位）")
    rows = []
    for scope_key, label in (("all", "全体"), ("pdf_only", "仅 PDF"), ("non_pdf", "非 PDF")):
        blk = ln.get(scope_key) or {}
        if not blk or not blk.get("count"):
            continue
        q = blk.get("quantiles") or {}
        rows.append([label, f"{blk['count']:,}", str(blk.get("min", "-")),
                     str(blk.get("mean", "-"))[:6], str(q.get("P50", "-")),
                     str(q.get("P90", "-")), str(blk.get("max", "-"))])
    if rows:
        p3.table_block(["范围", "份数", "最小", "均值", "P50", "P90", "最大"], rows,
                       widths=[2, 2, 2, 2, 2, 2, 2])
    else:
        p3.note("（该报告未提供分位统计）")
    p3.section("2. 长度直方图（全体，节选前 8 档）")
    hist = (ln.get("all") or {}).get("histogram") or []
    p3.table_block(["区间", "份数", "占比"],
                   [[h["label"], f"{h['count']:,}", f"{h['ratio'] * 100:.1f}%"] for h in hist[:8]],
                   widths=[3, 2, 2])
    p3.section("3. 阈值配置")
    th = ln.get("thresholds") or {}
    p3.table_block(["阈值项", "值"], [[k, str(v)] for k, v in list(th.items())[:8]], widths=[4, 2])
    p3.finish(SHOT / "文档长度分布_按类型.png", sources=[SRC])

    # ------------------------------------------------ 图4 敏感与待审
    p4 = Page("工单18 敏感信息检测与待审清单", f"敏感规则 {len(sens.get('enabled_rules') or [])} 条启用")
    p4.section("1. 敏感信息检测")
    p4.kv_block([
        ("是否启用", str(sens["enabled"]).lower()),
        ("启用规则数", str(len(sens.get("enabled_rules") or []))),
        ("命中文件数", str(sens["files_with_hits"])),
        ("命中总次数", str(sens["total_hits"])),
        ("待人工复核", str(sens["pending_review"])),
    ], cols=2)
    p4.section("2. 待审清单（三大类）")
    p4.table_block(["清单", "条数"], [
        ["PDF 类型待确认", str(pend.get("pdf_type_pending_confirm_total", "-"))],
        ["版本冲突待处理", str(pend.get("version_conflicts_total", "-"))],
        ["敏感信息待复核", str(pend.get("sensitive_pending_review_total", "-"))],
        ["损坏文件", str(pend.get("corrupted_files_total", "-"))],
    ], widths=[4, 2])
    p4.section("3. 路由规则")
    rr = d.get("routing_rules") or {}
    p4.kv_block([("默认解析器", str(rr.get("default_parser"))),
                 ("降级解析器", str(rr.get("degraded_parsers")))[:60],
                 ("规则条数", str(len(rr.get("rules") or [])))], cols=1)
    p4.finish(SHOT / "敏感信息检测与待审清单.png", sources=[SRC])

    print("[OK] 4 张截图 ->", SHOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
