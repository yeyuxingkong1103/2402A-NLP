# -*- coding: utf-8 -*-
# agent_workflow.py 工单18 文档入库智能体工作流
from document_quality_assessment import DocumentQualityAssessmentSkill

def document_ingestion_workflow(doc_dir:str):
    """
    文档入库工作流：
    1.调用文档质量评估Skill
    2.读取每个文档分类标签
    3.按标签路由分发不同解析任务
    """
    skill = DocumentQualityAssessmentSkill()
    skill_out = skill.run(doc_dir)
    report = skill_out["quality_report"]
    doc_label_map = report["document_label"]

    routing_result = []
    for file_path,label in doc_label_map.items():
        handler = None
        if label == "Clean_Markdown":
            handler = "direct_chunk_ingest"
        elif label == "Scan_PDF":
            handler = "ocr_pdf_parser"
        elif label == "Table_Heavy":
            handler = "table_special_parser"
        elif label == "Image_Heavy":
            handler = "multimodal_figure_parser"
        elif label == "Parse_Failed":
            handler = "manual_review"
        routing_result.append({
            "file":file_path,
            "label":label,
            "dispatch_handler":handler
        })
    return {
        "quality_report":report,
        "routing_assign":routing_result
    }
