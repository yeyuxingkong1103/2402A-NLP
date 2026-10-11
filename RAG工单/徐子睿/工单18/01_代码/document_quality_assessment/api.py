# -*- coding: utf-8 -*-
"""
DocumentQualityAssessmentSkill · API 端点
POST /v1/document/quality-inspection
  入参: {"path": "<文件夹路径>"} 或 {"files": ["...", "..."]}
  出参: {"code":0,"data":{...结构化报告...},"html":"<简报 HTML>"}

集成到 RAGFlow：把 build_app() 的 APIRouter/Blueprint 注册进 RAGFlow 的 api/apps。
"""
from __future__ import annotations

import os

from . import Config, assess, report as R

try:
    from fastapi import FastAPI, Body
    from fastapi.responses import JSONResponse, HTMLResponse
    _HAS_FASTAPI = True
except Exception:
    _HAS_FASTAPI = False


def run_inspection(payload: dict) -> dict:
    root = payload.get("path")
    cfg = Config.load(payload.get("config"))
    files = payload.get("files")
    if files:
        # 支持文件列表：临时目录不变，直接对共同父目录评估后过滤也可；
        # 简化：对父目录评估
        root = root or os.path.dirname(files[0])
    if not root or not os.path.isdir(root):
        return {"code": 400, "message": f"invalid path: {root}"}
    rep = assess(root, cfg, limit=int(payload.get("limit", 0)))
    html = R.to_html(rep)
    return {"code": 0, "data": rep, "html": html}


def build_app():
    if not _HAS_FASTAPI:
        raise RuntimeError("fastapi 未安装：pip install fastapi uvicorn")
    app = FastAPI(title="DocumentQualityAssessmentSkill API")

    @app.post("/v1/document/quality-inspection")
    async def quality_inspection(payload: dict = Body(...)):
        res = run_inspection(payload)
        if payload.get("format") == "html" and res.get("code") == 0:
            return HTMLResponse(res["html"])
        return JSONResponse(res)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=9380)
