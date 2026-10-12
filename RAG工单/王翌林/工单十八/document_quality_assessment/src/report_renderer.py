# -*- coding: utf-8 -*-
"""将质检报告渲染为 HTML 简报（重点呈现待确认/待审核列表）。"""
import os
import html


def render_html(report):
    fd = report["format_distribution"]
    pdfs = report["pdf_classification"]
    ld = report["length_distribution"]
    dups = report["duplicates"]
    sens = report["sensitive_info"]
    ai = report["action_items"]

    rows = []
    for ext, v in fd["distribution"].items():
        rows.append(f"<tr><td>{html.escape(ext)}</td><td>{v['count']}</td><td>{v['ratio']}%</td></tr>")
    fmt_tbl = "".join(rows)

    pdf_rows = "".join(
        f"<tr><td>{html.escape(os.path.basename(p['file']))}</td><td>{p['label']}</td>"
        f"<td>{p['pages']}</td><td>{p.get('scan_ratio','')}</td></tr>"
        for p in pdfs["pdfs"]
    )

    conf = "".join(
        f"<li>{html.escape(x['file'])} — {html.escape(x['reason'])}</li>"
        for x in ai["to_confirm_pdf"]
    ) or "<li>无</li>"

    vers = "".join(
        f"<li>{html.escape(os.path.basename(x['file_a']))} ↔ {html.escape(os.path.basename(x['file_b']))}"
        f"（汉明距离 {x['hamming_distance']}）<br><small>片段A: {html.escape(x['snippet_a'])}<br>"
        f"片段B: {html.escape(x['snippet_b'])}</small></li>"
        for x in ai["version_conflicts"]
    ) or "<li>无</li>"

    exact = "".join(
        f"<li>{' | '.join(html.escape(os.path.basename(f)) for f in g)}</li>"
        for g in dups["exact_duplicates"]
    ) or "<li>无</li>"

    sens_rows = "".join(
        f"<tr><td>{html.escape(os.path.basename(x['file']))}</td><td>{x['type']}</td>"
        f"<td>{html.escape(x['value'])}</td><td>{html.escape(x['context'])}</td></tr>"
        for x in sens["to_review"]
    ) or "<tr><td colspan=4>无</td></tr>"

    pct = " ".join(f"{k}={v}" for k, v in ld["percentiles"].items())
    bkt = " ".join(f"{k}:{v}" for k, v in ld["buckets"].items())

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>文档质量评估报告</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px}}h2{{color:#2c5282}}</style></head><body>
<h1>文档质量评估报告</h1>
<p>输入：{html.escape(str(report['input']))}　文件数：{report['file_count']}</p>
<h2>1. 格式分布</h2><table><tr><th>格式</th><th>数量</th><th>占比</th></tr>{fmt_tbl}</table>
<h2>2. PDF 分类</h2><table><tr><th>文件</th><th>标签</th><th>页数</th><th>扫描页占比</th></tr>{pdf_rows}</table>
<h3>待确认（PDF）</h3><ul>{conf}</ul>
<h2>3. 文档长度分布</h2><p>分位数：{pct}</p><p>区间分布：{bkt}</p>
<h2>4. 重复检测</h2>
<h3>精确重复（MD5）</h3><ul>{exact}</ul>
<h3>待确认版本冲突（SimHash）</h3><ul>{vers}</ul>
<h2>5. 敏感信息（待审核）</h2>
<table><tr><th>文件</th><th>类型</th><th>命中</th><th>上下文</th></tr>{sens_rows}</table>
</body></html>"""
