# -*- coding: utf-8 -*-
"""DocumentQualityAssessmentSkill · 报告生成（JSON 结构 + HTML 简报）"""
from __future__ import annotations

import html
import json
from datetime import datetime


def to_json(report: dict, path: str | None = None) -> str:
    s = json.dumps(report, ensure_ascii=False, indent=2)
    if path:
        open(path, "w", encoding="utf-8").write(s)
    return s


def to_html(report: dict, path: str | None = None) -> str:
    meta = report.get("meta", {})
    fs = report.get("format_stats", {})
    pdf = report.get("pdf_types", {})
    ld = report.get("length_distribution", {})
    dup = report.get("duplicates", {})
    sen = report.get("sensitive", {})
    cls = report.get("classification", {})

    def rows(items, cols):
        out = []
        for it in items[:200]:
            out.append("<tr>" + "".join(f"<td>{html.escape(str(it.get(c,'')))}</td>" for c in cols) + "</tr>")
        return "\n".join(out)

    ext_rows = rows([{"ext": k, **v} for k, v in fs.get("by_ext", {}).items()], ["ext", "count", "ratio"])

    css = """<style>body{font-family:system-ui,'Microsoft YaHei';max-width:1000px;margin:24px auto;color:#222}
h1{font-size:22px}h2{font-size:17px;border-left:4px solid #4a6cf7;padding-left:8px;margin-top:28px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:6px 8px;text-align:left}
th{background:#f5f7ff}.k{color:#4a6cf7;font-weight:600}.warn{color:#c0392b}</style>"""

    h = [f"<html><head><meta charset='utf-8'>{css}</head><body>"]
    h.append("<h1>文档质量评估简报（DocumentQualityAssessmentSkill）</h1>")
    h.append(f"<p>扫描目录：<b>{html.escape(str(meta.get('root','')))}</b>　"
             f"文件数：<b>{meta.get('files',0)}</b>　生成时间：{html.escape(str(report.get('generated_at','')))}</p>")

    h.append("<h2>1. 格式分布</h2><table><tr><th>扩展名</th><th>数量</th><th>占比</th></tr>"
             f"{ext_rows}</table>")

    h.append("<h2>2. PDF 页面类型识别</h2>")
    h.append(f"<p>PDF 总数：<b>{pdf.get('pdf_total',0)}</b>　"
             f"扫描型 <span class='warn'>{pdf.get('scan',0)}</span>　"
             f"文字型 {pdf.get('text',0)}　混合型 {pdf.get('mixed',0)}</p>")
    conf = pdf.get("待确认", [])
    if conf:
        h.append(f"<p class='warn'>待确认（无法打开/空文档）：{len(conf)}</p><table>"
                 "<tr><th>文件</th><th>说明</th></tr>" +
                 rows([{"file": c.get("file"), "note": c.get("error", c.get("type", ""))} for c in conf],
                      ["file", "note"]) + "</table>")

    h.append("<h2>3. 文档长度分布</h2>")
    h.append("<table><tr><th>指标</th><th>值</th></tr>" +
             "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in ld.items() if k != "bins") + "</table>")
    if ld.get("bins"):
        h.append("<table><tr><th>区间(字符)</th><th>文档数</th></tr>" +
                 "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in ld["bins"].items()) + "</table>")

    h.append("<h2>4. 重复检测</h2>")
    h.append(f"<p>MD5 完全相同分组：<b>{len(dup.get('md5',[]))}</b>　"
             f"SimHash 高相似(待确认版本冲突)：<b class='warn'>{len(dup.get('near',[]))}</b></p>")
    if dup.get("md5"):
        h.append("<table><tr><th>MD5</th><th>文件</th></tr>" +
                 rows([{"md5": d["md5"][:12] + "…", "files": "<br>".join(d["files"])} for d in dup["md5"]],
                      ["md5", "files"]) + "</table>")
    if dup.get("near"):
        h.append("<table><tr><th>文件A</th><th>文件B</th><th>汉明距离</th><th>状态</th></tr>" +
                 rows(dup["near"], ["file_a", "file_b", "hamming", "status"]) + "</table>")

    h.append("<h2>5. 敏感信息检测（待审核）</h2>")
    h.append(f"<p>命中统计：{html.escape(json.dumps(sen.get('counts',{}), ensure_ascii=False))}　"
             f"（银行卡检测：{'开' if sen.get('enabled',{}).get('bankcard') else '默认关闭'}）</p>")
    if sen.get("items"):
        h.append("<table><tr><th>文件</th><th>类型</th><th>命中</th><th>上下文</th></tr>" +
                 rows(sen["items"], ["file", "kind", "value", "context"]) + "</table>")

    h.append("<h2>6. 文档分类标签</h2>")
    h.append("<table><tr><th>标签</th><th>文档数</th></tr>" +
             "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in cls.get("label_counts", {}).items()) +
             "</table>")

    h.append("</body></html>")
    doc = "\n".join(h)
    if path:
        open(path, "w", encoding="utf-8").write(doc)
    return doc
