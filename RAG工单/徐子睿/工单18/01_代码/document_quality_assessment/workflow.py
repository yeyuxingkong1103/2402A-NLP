# -*- coding: utf-8 -*-
"""
DocumentQualityAssessmentSkill · 智能体工作流集成
document_ingestion_workflow：触发质检 → 调用 Skill → 按分类标签路由到不同解析器
（扫描型 → OCR 解析器；文字型 → 直接文本解析；混合型 → 先 OCR 后补文本）
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .config import Config
from .skill import assess


# 标签 → 解析器路由表（可外置配置）
ROUTING = {
    "Scan_PDF": "ocr_parser",       # 扫描型 → OCR
    "Text_PDF": "plain_parser",     # 文字型 → 直接抽取
    "Mixed_PDF": "hybrid_parser",   # 混合型 → OCR + 文本
    "Patent": "deepdoc_parser",     # 专利 → DeepDOC
}


def route_by_label(label: str) -> str:
    """按分类标签选择解析器（含专利标签优先）。"""
    for tag, parser in ROUTING.items():
        if tag in label:
            return parser
    return "default_parser"


def document_ingestion_workflow(root: str,
                                cfg: Config | None = None,
                                parser_dispatch: Dict[str, Callable] | None = None,
                                limit: int = 0) -> dict:
    """
    1) 触发质检（调用 DocumentQualityAssessmentSkill）
    2) 依据分类标签，把文档路由到对应解析器
    返回 {'report': ..., 'routing': {file: parser}, 'dispatch_result': ...}
    """
    cfg = cfg or Config()
    report = assess(root, cfg, limit=limit)

    # 依据分类标签逐文件路由
    from . import detectors as D
    routing: Dict[str, str] = {}
    files = list(D.iter_files(root, cfg.extensions))
    if limit:
        files = files[:limit]
    for f in files:
        ext = f.lower().rsplit(".", 1)[-1]
        pt = D.pdf_page_type(f, cfg) if ext == "pdf" else None
        label = D.classify_doc(f, (pt or {}).get("type"), len(D.read_text(f)))
        routing[f] = route_by_label(label)

    dispatch_result = {}
    if parser_dispatch:
        for f, parser_name in routing.items():
            fn = parser_dispatch.get(parser_name)
            if fn:
                dispatch_result[f] = fn(f)

    return {"report": report, "routing": routing, "dispatch_result": dispatch_result}
