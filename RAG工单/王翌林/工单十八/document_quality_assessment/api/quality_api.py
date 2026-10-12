# -*- coding: utf-8 -*-
"""
质检 API 端点：POST /v1/document/quality-inspection
接收文件夹路径或文件列表，触发质检 Skill，返回 JSON 及 HTML 简报。
运行：PYTHONPATH=. uvicorn api.quality_api:app --port 8018
"""
import os, sys, json
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quality_assessor import assess, load_config
from src.report_renderer import render_html

app = FastAPI(title="文档质量评估 API")


class InspectReq(BaseModel):
    path: Optional[str] = None          # 文件夹路径
    files: Optional[List[str]] = None   # 或文件列表
    return_html: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "skill": "DocumentQualityAssessmentSkill"}


@app.post("/v1/document/quality-inspection")
def inspect(req: InspectReq):
    if req.path:
        report = assess(req.path)
    elif req.files:
        # 对文件列表：逐文件评估后合并（此处直接取首个所在目录的简化实现）
        cfg = load_config()
        texts = {}
        from src.quality_assessor import _read_text
        for f in req.files:
            ext = os.path.splitext(f)[1].lower()
            t, _ = _read_text(f, ext); texts[f] = t
        report = assess(req.files[0], cfg) if req.files else {}
    else:
        return JSONResponse({"error": "需提供 path 或 files"}, status_code=400)

    result = {"report": report}
    if req.return_html:
        result["html"] = render_html(report)
    return result


@app.get("/v1/document/quality-inspection/html", response_class=HTMLResponse)
def inspect_html(path: str):
    return render_html(assess(path))
