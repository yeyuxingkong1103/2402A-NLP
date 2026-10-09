import sys, os
sys.path.insert(0, "/root/autodl-tmp/rag_project/document_quality_assessment")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from main import assess, load_config
from utils.report_utils import to_html_brief

router = APIRouter()

class QualityRequest(BaseModel):
    folder_path: Optional[str] = None
    file_list: Optional[List[str]] = None
    config_path: str = "/root/autodl-tmp/rag_project/document_quality_assessment/assessment_config.yaml"

@router.post("/v1/document/quality-inspection")
def quality_inspection(req: QualityRequest):
    if not req.folder_path and not req.file_list:
        raise HTTPException(400, "folder_path 或 file_list 必填")
    folder = req.folder_path or "/root/autodl-tmp/rag_project/data/IMDR/documents"
    try:
        report = assess(folder, req.config_path)
    except Exception as e:
        raise HTTPException(500, "评估失败: %s" % e)
    return {
        "json_report": report,
        "html_brief": to_html_brief(report),
    }

@router.get("/v1/document/health")
def health():
    return {"status": "ok", "skill": "DocumentQualityAssessmentSkill"}
