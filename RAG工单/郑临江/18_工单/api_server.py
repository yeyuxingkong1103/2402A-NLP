# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流工单
质检 API 端点：POST /v1/document/quality-inspection
接收文件夹路径或文件列表，触发质检 Skill，返回 JSON 报告与 HTML 简报。
"""
import os

from flask import Flask, jsonify, request

from document_quality_assessment import DocumentQualityAssessor

app = Flask(__name__)


def render_html(report):
    """生成 HTML 简报，重点展示待确认/待审核列表。"""
    dup = report["duplicates"]
    sens = report["sensitive_info"]
    scanned = {k: v for k, v in report["pdf_types"].items()
               if v["type"] in ("Scan_PDF", "Mixed_PDF")}
    rows_dup = "".join(
        f"<li>{', '.join(d['files'])}（{'完全重复' if d['kind']=='exact' else '版本冲突,距离'+str(d.get('distance',''))}）</li>"
        for d in dup["version_conflicts"] + dup["exact"]
    )
    rows_sens = "".join(
        f"<li>{s['file']} -> {s['type']}: {s['value']}（上下文: {s['context']}）</li>"
        for s in sens
    )
    rows_scan = "".join(f"<li>{k}（扫描页 {v['scanned_pages']}）</li>" for k, v in scanned.items())
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>文档质检报告</title></head>
<body><h1>文档质量评估报告</h1>
<p>共 {report['total_files']} 个文件</p>
<h2>格式分布</h2><pre>{report['format_distribution']}</pre>
<h2>扫描型/混合型 PDF</h2><ul>{rows_scan or '<li>无</li>'}</ul>
<h2>待确认的版本冲突列表</h2><ul>{rows_dup or '<li>无</li>'}</ul>
<h2>待审核的敏感信息列表</h2><ul>{rows_sens or '<li>无</li>'}</ul>
</body></html>"""


@app.post("/v1/document/quality-inspection")
def quality_inspection():
    body = request.get_json(force=True, silent=True) or {}
    folder = body.get("folder")
    files = body.get("files")

    assessor = DocumentQualityAssessor()
    if files:
        report = assessor.assess_files(files)
    elif folder and os.path.isdir(folder):
        report = assessor.assess(folder)
    else:
        return jsonify({"error": "请提供 folder 路径或 files 列表"}), 400

    want_html = request.args.get("format") == "html" or body.get("format") == "html"
    if want_html:
        return render_html(report), 200, {"Content-Type": "text/html; charset=utf-8"}
    return jsonify(report)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
