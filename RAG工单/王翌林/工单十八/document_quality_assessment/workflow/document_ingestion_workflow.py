# -*- coding: utf-8 -*-
"""
智能体工作流：document_ingestion_workflow
核心逻辑：触发质检 -> 调用 DocumentQualityAssessmentSkill -> 根据分类标签路由到不同解析器
（扫描型路由至 OCR 解析器）。
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quality_assessor import assess, load_config


# 模拟解析器（对应 route_map）
PARSERS = {
    "text_extractor": "直接提取文本",
    "ocr_parser": "调用 OCR 解析扫描页",
    "office_extractor": "解析 Office 文档",
    "markdown_extractor": "解析 Markdown",
    "fallback_parser": "兜底人工处理",
}


def run_workflow(input_path, cfg=None):
    """执行入库工作流，返回每步决策记录。"""
    cfg = cfg or load_config()
    # 节点1：触发质检
    report = assess(input_path, cfg)
    # 节点2：按分类标签路由
    routing = []
    for f, info in report["classification"].items():
        routing.append({
            "file": f,
            "label": info["label"],
            "route_to": info["route_to"],
            "action": PARSERS.get(info["route_to"], "未知解析器"),
            "needs_review": info["label"] in ("Scan_PDF", "Mixed_PDF", "Unknown"),
        })
    # 节点3：汇总需人工介入项
    human_review = {
        "scan_pdfs": [f for f, i in report["classification"].items() if i["label"] in ("Scan_PDF", "Mixed_PDF")],
        "exact_duplicates": report["duplicates"]["exact_duplicates"],
        "version_conflicts": report["action_items"]["version_conflicts"],
        "sensitive": report["action_items"]["sensitive_to_review"],
    }
    return {"workflow": "document_ingestion_workflow",
            "quality_report": report, "routing": routing, "human_review": human_review}


if __name__ == "__main__":
    res = run_workflow(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(json.dumps({"workflow": res["workflow"],
                      "routing_count": len(res["routing"]),
                      "scan_pdfs": len(res["human_review"]["scan_pdfs"])}, ensure_ascii=False, indent=2))
