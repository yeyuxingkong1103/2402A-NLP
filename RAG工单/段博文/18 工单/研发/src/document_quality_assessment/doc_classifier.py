# -*- coding: utf-8 -*-
"""文档分类标签体系：格式标签 + 全局补充（长度/风险）标签，并给出解析路由。"""
from __future__ import annotations

from typing import Dict, List

# 格式维度标签 -> 默认路由
_FORMAT_ROUTE = {
    "Text_PDF": "DirectParser",
    "Hybrid_PDF": "HybridParser",
    "Scan_PDF": "OCRParser",
    "Corrupt_PDF": "ReviewNode",
    "DOCX_Doc": "DirectParser",
    "Markdown_Doc": "DirectParser",
    "Text_Doc": "DirectParser",
    "Unknown_Format": "ReviewNode",
}


def format_tag_for(path, pdf_info: Dict = None, read_ok: bool = True) -> str:
    """根据文件类型/读取结果给出格式标签。"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        if not read_ok or (pdf_info and pdf_info["type"] == "Corrupt_PDF"):
            return "Corrupt_PDF"
        return pdf_info["type"] if pdf_info else "Text_PDF"
    if ext in (".docx", ".doc"):
        return "DOCX_Doc" if read_ok else "Corrupt_File"
    if ext == ".md":
        return "Markdown_Doc"
    if ext == ".txt":
        return "Text_Doc"
    return "Unknown_Format"


def route_for(tags: List[str]) -> str:
    """根据标签集合决定工作流路由（取最高优先级）。

    优先级：损坏/风险审核 > OCR > 混合解析 > 直接解析。
    """
    tag_set = set(tags)
    if {"Corrupt_PDF", "Corrupt_File", "Empty_Doc", "Unknown_Format"} & tag_set:
        return "ReviewNode"
    if "Sensitive_Info" in tag_set:
        return "SecurityReviewNode"
    if "Near_Duplicate" in tag_set:
        return "VersionReviewNode"
    if "Exact_Duplicate" in tag_set:
        return "DedupeNode"
    if "Scan_PDF" in tag_set:
        return "OCRParser"
    if "Hybrid_PDF" in tag_set:
        return "HybridParser"
    return "DirectParser"


def apply_global_tags(records: List[Dict], md5_group_map: Dict[str, List[str]],
                      simhash_paths: set, length_stats: Dict,
                      length_config: dict, tags_config: dict) -> None:
    """全局信息出来后，就地给每条记录补充长度/风险标签并计算最终路由。

    - length_config: config["length"]（含 empty_threshold）
    - tags_config: config["tags"]（标签开关）
    - md5_group_map: 出现在重复组中的 path（重复组中第一个保留，其余 Exact_Duplicate）
    - simhash_paths: 出现在 simhash 待确认对中的 path
    """
    empty_thr = length_config["empty_threshold"]
    p25 = length_stats["percentiles"].get("P25", 0)
    p90 = length_stats["percentiles"].get("P90", 0)
    enable_length = tags_config.get("enable_length_tags", True)
    enable_risk = tags_config.get("enable_risk_tags", True)

    for rec in records:
        tags = list(rec["tags"])
        cc = rec["char_count"]
        # 纯扫描型 PDF 的空文本层是预期现象，不参与长度标签
        is_scan_pdf = "Scan_PDF" in tags
        if enable_length and not is_scan_pdf:
            if cc <= empty_thr:
                tags.append("Empty_Doc")
            elif cc and cc <= p25:
                tags.append("Short_Doc")
            if cc and cc >= p90 and cc > empty_thr:
                tags.append("Long_Doc")
        if enable_risk:
            if rec["path"] in md5_group_map:
                tags.append("Exact_Duplicate")
            if rec["path"] in simhash_paths:
                tags.append("Near_Duplicate")
            if rec.get("sensitive_findings"):
                tags.append("Sensitive_Info")
        # 去重保序
        seen = set()
        rec["tags"] = [t for t in tags if not (t in seen or seen.add(t))]
        rec["route"] = route_for(rec["tags"])
