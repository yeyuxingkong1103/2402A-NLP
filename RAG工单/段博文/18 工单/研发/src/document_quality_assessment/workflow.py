# -*- coding: utf-8 -*-
"""document_ingestion_workflow：质检驱动的文档入库智能体工作流。

拓扑：
    [触发质检] -> [DocumentQualityAssessmentSkill]
        -> 决策节点(按 route 标签)
            -> DirectParser / OCRParser / HybridParser / DedupeNode
            -> SecurityReviewNode / VersionReviewNode / ReviewNode
"""
from __future__ import annotations

import time
from typing import Dict, List


# ---------------- 各处理节点（生产中替换为真实解析器实现）----------------
class DirectParser:
    name = "DirectParser"

    def run(self, rec: Dict) -> Dict:
        return {"parser": "直接文本解析", "chars": rec["char_count"], "status": "已入库"}


class OCRParser:
    name = "OCRParser"

    def run(self, rec: Dict) -> Dict:
        pages = rec["pdf"]["scanned_pages"] if rec.get("pdf") else 0
        return {"parser": "OCR 解析", "ocr_pages": pages,
                "steps": ["pdfium 渲染页面", "方向检测/旋转矫正", "OCR 识别", "版面还原"],
                "status": "OCR完成并入库"}


class HybridParser:
    name = "HybridParser"

    def run(self, rec: Dict) -> Dict:
        return {"parser": "混合解析", "text_pages": rec["pdf"].get("text_pages", 0),
                "ocr_pages": rec["pdf"].get("scanned_pages", 0), "status": "已入库"}


class DedupeNode:
    name = "DedupeNode"

    def run(self, rec: Dict) -> Dict:
        return {"action": "判定为重复副本", "status": "跳过（保留主文件）"}


class SecurityReviewNode:
    name = "SecurityReviewNode"

    def run(self, rec: Dict) -> Dict:
        return {"action": "敏感信息安全审核", "pending_items": len(rec["sensitive_findings"]),
                "status": "挂起，等待审核"}


class VersionReviewNode:
    name = "VersionReviewNode"

    def run(self, rec: Dict) -> Dict:
        return {"action": "版本冲突人工确认", "status": "挂起，等待确认"}


class ReviewNode:
    name = "ReviewNode"

    def run(self, rec: Dict) -> Dict:
        return {"action": "损坏/空文档人工处理", "error": rec.get("read_error"), "status": "挂起"}


_NODES = {c.name: c for c in (DirectParser, OCRParser, HybridParser, DedupeNode,
                              SecurityReviewNode, VersionReviewNode, ReviewNode)}


class DocumentIngestionWorkflow:
    """质检专用工作流。构造时传入一份质检报告（dict）。"""

    def __init__(self, quality_report: Dict):
        self.report = quality_report
        self._index = {r["path"]: r for r in quality_report["records"]}

    def run(self, path: str) -> Dict:
        """对单个文档执行完整工作流，返回含节点轨迹的结果。"""
        rec = self._index.get(str(path))
        if rec is None:
            return {"path": str(path), "error": "未在质检报告中找到该文件", "trace": []}

        trace = [
            {"node": "TriggerNode", "action": "触发文档入库工作流"},
            {"node": "DocumentQualityAssessmentSkill", "action": "读取质检标签",
             "tags": rec["tags"], "route": rec["route"]},
            {"node": "DecisionNode", "action": f"按标签路由 -> {rec['route']}"},
        ]
        result = _NODES[rec["route"]]().run(rec)
        trace.append({"node": rec["route"], "action": result})
        return {"path": rec["path"], "tags": rec["tags"], "route": rec["route"],
                "trace": trace, "result": result}

    def run_many(self, paths: List[str]) -> List[Dict]:
        return [self.run(p) for p in paths]
