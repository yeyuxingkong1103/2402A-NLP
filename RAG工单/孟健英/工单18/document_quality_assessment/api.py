# 工单编号：人工智能 NLP-RAG-实现文档质量评估 Skill 并集成至智能体工作流工单
"""
DocumentQualityAssessmentSkill - FastAPI 服务
暴露 /v1/document/quality-inspection 端点
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

from skill import DocumentQualityAssessmentSkill

app = FastAPI(title="Document Quality Assessment Skill API")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "assessment_config.yaml")
OUTPUT_DIR = Path("/mnt/d/专高六 工单/工单18/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class InspectRequest(BaseModel):
    folder_path: Optional[str] = None
    file_list: Optional[List[str]] = None
    max_files: Optional[int] = None


def render_html(report: dict) -> str:
    meta = report["meta"]
    fmt = report["format_distribution"]
    pdf = report["pdf_type_distribution"]["summary"]
    dup = report["duplicate_detection"]
    sens = report["sensitive_detection"]

    rows = "".join(
        f"<tr><td>{x['ext']}</td><td>{x['count']}</td><td>{x['ratio']:.2%}</td></tr>"
        for x in fmt["by_format"]
    )
    pdf_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in pdf.items())

    sens_rows = "".join(
        f"<tr><td>{s['file']}</td><td>{s['type']}</td><td>{s['match']}</td>"
        f"<td>{s['context'][:80]}</td></tr>"
        for s in sens["pending_review_list"][:50]
    )

    dup_rows = "".join(
        f"<tr><td>{d['md5'][:12]}...</td><td>{len(d['files'])}</td>"
        f"<td>{'<br>'.join(d['files'][:3])}</td></tr>"
        for d in dup["md5_duplicates"][:30]
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>质检报告</title>
<style>body{{font-family:sans-serif;margin:20px}}
table{{border-collapse:collapse;margin:10px 0;width:100%}}
th,td{{border:1px solid #ccc;padding:6px;text-align:left}}
th{{background:#f0f0f0}}h2{{margin-top:30px;color:#333}}</style>
</head><body>
<h1>文档质量评估报告</h1>
<p><b>目录</b>: {meta['folder']}</p>
<p><b>文件总数</b>: {meta['total_files']}</p>
<p><b>扫描时间</b>: {meta['scan_time']}</p>

<h2>1. 格式分布</h2>
<table><tr><th>扩展名</th><th>数量</th><th>占比</th></tr>{rows}</table>

<h2>2. PDF 类型分布</h2>
<table><tr><th>类型</th><th>数量</th></tr>{pdf_rows}</table>

<h2>3. 长度分布</h2>
<pre>{json.dumps(report['length_distribution']['percentiles'], indent=2)}</pre>

<h2>4. MD5 重复组（{dup['md5_duplicate_groups']}）</h2>
<table><tr><th>MD5</th><th>数量</th><th>文件</th></tr>{dup_rows or '<tr><td colspan=3>无</td></tr>'}</table>

<h2>5. 敏感信息待审核（{sens['total']}）</h2>
<table><tr><th>文件</th><th>类型</th><th>匹配</th><th>上下文</th></tr>{sens_rows or '<tr><td colspan=4>无</td></tr>'}</table>
</body></html>"""


@app.post("/v1/document/quality-inspection")
async def quality_inspection(req: InspectRequest):
    try:
        skill = DocumentQualityAssessmentSkill(CONFIG_PATH)
        folder = req.folder_path or "/mnt/d/专高六 工单/工单18/data/documents"
        report = skill.assess_folder(folder, max_files=req.max_files)

        json_path = OUTPUT_DIR / "quality_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        html_path = OUTPUT_DIR / "quality_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(render_html(report))

        return JSONResponse({
            "status": "ok",
            "json_report": str(json_path),
            "html_report": str(html_path),
            "summary": {
                "total_files": report["meta"]["total_files"],
                "pdf_types": report["pdf_type_distribution"]["summary"],
                "md5_duplicates": report["duplicate_detection"]["md5_duplicate_groups"],
                "sensitive_total": report["sensitive_detection"]["total"],
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/document/quality-inspection/report", response_class=HTMLResponse)
async def get_html_report():
    html_path = OUTPUT_DIR / "quality_report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在，请先 POST 触发质检")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
