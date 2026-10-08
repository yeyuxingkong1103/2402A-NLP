# -*- coding: utf-8 -*-
"""HTML 简报生成：突出三类待办，敏感信息打码。"""
from __future__ import annotations

import html as _html
from typing import Dict


def _mask(value: str) -> str:
    if len(value) <= 6:
        return value[:2] + "****"
    return value[:3] + "****" + value[-2:]


def render_html(report: Dict) -> str:
    s = report["summary"]
    al = report["action_lists"]
    length = report["length"]
    e = _html.escape

    # ---- 顶部摘要卡 ----
    cards = "".join(
        f'<div class="card"><div class="num">{v}</div><div class="lab">{e(k)}</div></div>'
        for k, v in [
            ("文件总数", s["total_files"]),
            ("总大小(MB)", s["total_size_mb"]),
            ("扫描型PDF", s["pdf_type_counts"].get("Scan_PDF", 0)),
            ("MD5重复文件", s["md5_duplicate_files"]),
            ("版本冲突待确认", s["simhash_pending_pairs"]),
            ("敏感信息待审核", s["sensitive_pending_items"]),
            ("损坏文件", len(al["corrupt_files"])),
            ("空文档", len(al["empty_files"])),
        ])

    # ---- 待办1：扫描型待 OCR ----
    ocr_items = "".join(f"<li>{e(p)}</li>" for p in al["scan_pdf_for_ocr"][:200])
    more_ocr = f'<p class="more">…其余 {len(al["scan_pdf_for_ocr"]) - 200} 条见 JSON 报告</p>' if len(al["scan_pdf_for_ocr"]) > 200 else ""

    # ---- 待办2：版本冲突 ----
    vc_rows = "".join(
        f"<tr><td>{e(p['file_a'])}</td><td>{e(p['file_b'])}</td>"
        f"<td>{p['hamming_distance']}</td><td class='snip'>{e(p['similar_snippet'])}</td></tr>"
        for p in al["pending_version_conflicts"])
    vc_tbl = (
        "<table><tr><th>文件A</th><th>文件B</th><th>汉明距离</th><th>相似片段</th></tr>"
        f"{vc_rows}</table>") if al["pending_version_conflicts"] else '<p class="ok">无</p>'

    # ---- 待办3：敏感信息 ----
    mask_on = report["config_used"]["sensitive"]["mask_in_html"]
    se_rows = "".join(
        f"<tr><td>{e(f['file'])}</td><td>{f['type']}</td>"
        f"<td>{e(_mask(f['match']) if mask_on else f['match'])}</td>"
        f"<td class='snip'>{e(_mask(f['context']) if mask_on and f['type'] in ('手机号','身份证','银行卡') else f['context'])}</td></tr>"
        for f in al["pending_sensitive_review"][:300])
    se_tbl = (
        "<table><tr><th>文件</th><th>类型</th><th>命中（打码）</th><th>上下文</th></tr>"
        f"{se_rows}</table>") if al["pending_sensitive_review"] else '<p class="ok">无</p>'

    # ---- 格式分布 ----
    fmt_rows = "".join(
        f"<tr><td>{e(x['format'])}</td><td>{x['count']}</td><td>{x['ratio'] * 100:.1f}%</td></tr>"
        for x in report["format"]["by_format"])

    # ---- PDF 类型 / 临界 ----
    pdf_rows = "".join(f"<tr><td>{e(k)}</td><td>{v}</td></tr>" for k, v in s["pdf_type_counts"].items())
    bd_rows = "".join(
        f"<tr><td>{e(p['path'])}</td><td>{p['scanned_ratio'] * 100:.0f}%</td><td>{p['page_count']}</td></tr>"
        for p in al["pending_pdf_boundary"])

    # ---- 长度 ----
    pct_rows = "".join(f"<tr><td>{e(k)}</td><td>{v:,}</td></tr>" for k, v in length["percentiles"].items())
    bin_rows = "".join(
        f"<tr><td>{e(x['range'])}</td><td>{x['count']}</td><td>{x['ratio'] * 100:.1f}%</td></tr>"
        for x in length["bins_distribution"])

    # ---- MD5 ----
    md5_rows = "".join(
        f"<tr><td>{g['md5']}</td><td>{g['count']}</td>"
        f"<td class='snip'>{e('<br>'.join(g['files']))}</td></tr>"
        for g in report["duplicates"]["md5_groups"])
    md5_tbl = (
        "<table><tr><th>MD5</th><th>份数</th><th>文件</th></tr>"
        f"{md5_rows}</table>") if report["duplicates"]["md5_groups"] else '<p class="ok">无</p>'

    # ---- 路由统计 ----
    route_rows = "".join(f"<tr><td>{e(k)}</td><td>{v}</td></tr>" for k, v in s["routing_counts"].items())

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>文档质量评估简报</title>
<style>
 body{{font-family:"Microsoft YaHei",sans-serif;background:#0f1419;color:#d4d4d4;margin:0;padding:24px;}}
 h1{{font-size:22px}} h2{{font-size:16px;color:#4a90d9;border-left:4px solid #4a90d9;padding-left:8px;margin-top:32px}}
 .meta{{color:#888;font-size:12px}}
 .cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}
 .card{{background:#1a1f26;border:1px solid #2a3138;border-radius:8px;padding:12px 18px;min-width:110px;text-align:center}}
 .num{{font-size:22px;font-weight:bold;color:#5cb85c}} .lab{{font-size:12px;color:#888;margin-top:4px}}
 table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:12px}}
 th,td{{border:1px solid #2a3138;padding:6px 8px;text-align:left;vertical-align:top}}
 th{{background:#1a1f26}} tr:nth-child(even){{background:#14181d}}
 .snip{{max-width:420px;word-break:break-all;color:#9aa;font-size:11px}}
 .ok{{color:#5cb85c}} .more{{color:#888;font-size:11px}} ul{{font-size:12px;line-height:1.8}}
 .todo{{background:#241a1a;border:1px solid #5a2a2a;border-radius:8px;padding:4px 16px 12px}}
</style></head><body>
<h1>文档质量评估简报（DocumentQualityAssessmentSkill）</h1>
<div class="meta">目标：{e(report["target"])} ｜ 生成时间：{report["generated_at"]} ｜ 耗时：{report.get("elapsed_seconds","?")}s ｜ Skill v{report["version"]}</div>
<div class="cards">{cards}</div>

<div class="todo"><h2 style="color:#d9534f;border-color:#d9534f">待办1｜扫描型 PDF 待 OCR（{len(al["scan_pdf_for_ocr"])}）</h2>
<ul>{ocr_items}</ul>{more_ocr}</div>

<div class="todo"><h2 style="color:#f0ad4e;border-color:#f0ad4e">待办2｜版本冲突待确认（{len(al["pending_version_conflicts"])}）</h2>{vc_tbl}</div>

<div class="todo"><h2 style="color:#f0ad4e;border-color:#f0ad4e">待办3｜敏感信息待审核（{len(al["pending_sensitive_review"])}）</h2>{se_tbl}</div>

<h2>格式分布</h2><table><tr><th>格式</th><th>数量</th><th>占比</th></tr>{fmt_rows}</table>

<h2>PDF 类型分流</h2><table><tr><th>类型</th><th>数量</th></tr>{pdf_rows}</table>
<h2>PDF 临界样本（启发式待确认）</h2>
<table><tr><th>文件</th><th>扫描页占比</th><th>页数</th></tr>{bd_rows}</table>

<h2>文档长度分位数（字符）</h2><table>{pct_rows}</table>
<h2>长度区间分布</h2><table><tr><th>区间</th><th>数量</th><th>占比</th></tr>{bin_rows}</table>
<p>空文档（≤{length["empty_docs"] and report["config_used"]["length"]["empty_threshold"]} 字符）：{length["empty_docs"]} ｜ 平均：{length["mean_chars"]:,} 字符</p>

<h2>MD5 完全重复</h2>{md5_tbl}

<h2>解析路由统计（工作流决策结果）</h2><table><tr><th>路由节点</th><th>文档数</th></tr>{route_rows}</table>
</body></html>"""
