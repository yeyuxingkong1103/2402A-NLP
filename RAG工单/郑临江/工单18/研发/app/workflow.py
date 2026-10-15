"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。"""

from pathlib import Path
from document_quality_assessment import assess

REGISTRY = {"DocumentQualityAssessmentSkill": assess}


def document_ingestion_workflow(paths, config, output):
    report = REGISTRY["DocumentQualityAssessmentSkill"](paths, config, output)
    routes = [
        {
            "path": row["path"],
            "labels": row["labels"],
            "next_node": {
                "OCR": "OCRParser",
                "PAGE_LEVEL": "PageTypeRouter",
                "TEXT": "TextParser",
            }.get(row["route"], "ManualReview"),
        }
        for row in report["files"]
    ]
    return {
        "workflow": "document_ingestion_workflow",
        "skill": "DocumentQualityAssessmentSkill",
        "report": report,
        "routes": routes,
        "execution": "质检与路由决策已执行；未自动对全库触发耗时OCR任务",
    }
