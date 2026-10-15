# -*- coding: utf-8 -*-
"""
DQA · 文档分类标签体系

标签分两类：
  A. 处理路由标签（决定走哪个解析器）：
     Scan_PDF        → 扫描型PDF，需先 OCR
     Mixed_PDF       → 混合型，OCR + 文本双通道
     Text_PDF        → 文字型，直接文本抽取
     Structured_Table→ 表格类（xls/csv），走表格解析器
     Rule_Doc        → 法规/标准/专利类，按条款分块
     Long_Doc        → 超长文档，需分级/父子分块
  B. 风险/质量标签（提示人工处理）：
     Has_Sensitive   → 含敏感信息待审核
     Exact_Duplicate → MD5 完全重复
     Version_Conflict→ SimHash 高相似（待确认）
     Empty_Content   → 无有效文本（需检查解析器）
"""
from __future__ import annotations

from pathlib import Path

DOMAIN_RULES = [
    ("专利", ["专利", "CN10", "CN20", "申请公布号", "权利要求"]),
    ("年报/招股书", ["年报", "年度报告", "招股", "募集说明书", "600", "601", "000001"]),
    ("标准/规范", ["GB/T", "GB ", "国标", "技术规范", "标准", "规范"]),
    ("技术图纸/工艺", ["图纸", "工艺流程", "装配", "淬火", "公差", "工序"]),
    ("手册/说明书", ["说明书", "手册", "操作指南", "用户指南"]),
]


def _domain_tag(path: str, text_head: str = "") -> str:
    s = Path(path).stem + " " + text_head[:500]
    for tag, keys in DOMAIN_RULES:
        if any(k in s for k in keys):
            return tag
    return "通用文档"


def classify_documents(files: list, pdf_types: dict | None = None,
                       lengths: dict | None = None, dedup: dict | None = None,
                       sensitive: dict | None = None,
                       long_threshold: int = 50_000) -> dict:
    """
    files      : [{"path":..., "ext":...}]
    pdf_types  : classify_pdf_pages 的 _detail
    lengths    : analyze_length_distribution 的 _detail
    dedup      : detect_duplicates 结果
    sensitive  : detect_sensitive_info 结果
    """
    pdf_type_map = {d["file"]: d for d in (pdf_types or {}).get("_detail", [])}
    len_map = {d["file"]: d["chars"] for d in (lengths or {}).get("_detail", [])}

    exact_dup = set()
    for g in (dedup or {}).get("MD5重复组", []):
        for f in g["文件"][1:]:                 # 保留第一个，其余标重复
            exact_dup.add(f)
    ver_conf = set()
    for c in (dedup or {}).get("待确认的版本冲突列表", []):
        ver_conf.add(c["文件A"])
        ver_conf.add(c["文件B"])
    sens_files = {r["文件"] for r in (sensitive or {}).get("待审核敏感信息列表", [])}

    labels, unlabeled = [], 0
    for f in files:
        p = f["path"]
        ext = f["ext"]
        tags = []
        if ext == ".pdf":
            t = pdf_type_map.get(p, {}).get("type", "unknown")
            tags.append({"scanned": "Scan_PDF", "mixed": "Mixed_PDF",
                         "text": "Text_PDF"}.get(t, "Unknown_PDF"))
        elif ext in (".xlsx", ".xls", ".csv", ".tsv"):
            tags.append("Structured_Table")
        elif ext in (".docx", ".doc"):
            tags.append("Text_DOCX")
        else:
            tags.append("Other")

        n = len_map.get(p, 0)
        if n >= long_threshold:
            tags.append("Long_Doc")
        if n == 0:
            tags.append("Empty_Content")
        if p in exact_dup:
            tags.append("Exact_Duplicate")
        if p in ver_conf:
            tags.append("Version_Conflict")
        if p in sens_files:
            tags.append("Has_Sensitive")
        dom = _domain_tag(p)
        if dom != "通用文档":
            tags.append(dom)
        if not tags:
            unlabeled += 1

        labels.append({"file": p, "tags": tags, "domain": dom, "chars": n,
                       "route": _route(tags)})

    from collections import Counter
    tag_count = Counter(t for r in labels for t in r["tags"])
    return {
        "标注文件数": len(labels),
        "未标注文件数": unlabeled,
        "标签分布": dict(tag_count.most_common()),
        "需要OCR路由的文件数": tag_count.get("Scan_PDF", 0),
        "标签清单": labels,
        "字段说明": {
            "file": "文件路径", "tags": "标签数组", "domain": "领域标签",
            "route": "建议的解析路由", "chars": "字符数",
        },
    }


def _route(tags) -> str:
    if "Scan_PDF" in tags:
        return "OCR解析器（PaddleOCR / DeepDoc-OCR）"
    if "Structured_Table" in tags:
        return "表格解析器（RAGFlowExcelParser）"
    if "Empty_Content" in tags:
        return "人工检查/换解析器"
    if "Long_Doc" in tags:
        return "父子分块（Parent-Child）+ Raptor"
    return "默认文本解析器（naive）"
