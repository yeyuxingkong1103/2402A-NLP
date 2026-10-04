# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流工单
智能体工作流 document_ingestion_workflow：
触发质检 -> 调用 DocumentQualityAssessmentSkill -> 按分类标签路由到不同解析器
（如 Scan_PDF 路由至 OCR 解析器）。
"""
from document_quality_assessment import DocumentQualityAssessor


def route_by_label(label):
    """根据质检标签路由到对应解析器。"""
    routing = {
        "Scan_PDF": "ocr_parser",      # 扫描型 -> OCR 解析器
        "Mixed_PDF": "hybrid_parser",  # 混合型 -> 图文混合解析器
        "Text_PDF": "text_parser",     # 文字型 -> 直接文本解析器
        "Duplicate": "skip",           # 重复 -> 跳过
        "Version_Conflict": "manual_review",
        "Error": "error_handler",
        "Other": "default_parser",
    }
    return routing.get(label, "default_parser")


def document_ingestion_workflow(folder):
    """质检专用工作流：触发质检 -> 路由。"""
    assessor = DocumentQualityAssessor()
    report = assessor.assess(folder)

    routes = []
    for item in report["labels"]:
        parser = route_by_label(item["label"])
        routes.append({"file": item["file"], "label": item["label"], "parser": parser})
    report["routing"] = routes
    return report


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    result = document_ingestion_workflow(folder)
    print("质检完成，共", result["total_files"], "个文件")
    print("路由示例:", result["routing"][:5])
