# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程工单
查询理解与重写：自动识别问题中的视觉引用（“图3”“第11页图示”），
并构造图像增强查询，修复“检索无关”与“图文割裂”故障。
"""
import re

import config

FIGURE_PAT = re.compile(r"图\s*(\d+)")
PAGE_PAT = re.compile(r"第\s*(\d+)\s*页")
PART_PAT = re.compile(r"编号?\s*(\d+)\s*的部件|部件\s*(\d+)")


def detect_visual_reference(query):
    """检测视觉引用，返回 {'figure': n, 'page': n, 'part': n} 或 None。"""
    f = FIGURE_PAT.search(query)
    p = PAGE_PAT.search(query)
    part = PART_PAT.search(query)
    if not (f or p or part):
        return None
    return {
        "figure": int(f.group(1)) if f else None,
        "page": int(p.group(1)) if p else None,
        "part": int(part.group(1) or part.group(2)) if part else None,
    }


def is_visual_query(query):
    """是否含视觉引用关键词。"""
    return any(k in query for k in config.VISUAL_REF_HINTS)


def rewrite_query(query, figure_block=None):
    """查询重写：视觉引用 → 追加图像描述，构造图像增强查询。"""
    ref = detect_visual_reference(query)
    if not ref:
        return query
    if figure_block and figure_block.get("image_desc"):
        return f"{query} 【图像描述】{figure_block['image_desc']}"
    if ref["figure"]:
        return f"{query} 图{ref['figure']} 技术图纸 部件位置关系"
    if ref["part"] is not None:
        return f"{query} 部件{ref['part']} 位置 技术图纸"
    return query
