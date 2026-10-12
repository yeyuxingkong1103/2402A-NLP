# -*- coding: utf-8 -*-
"""
工单十五：查询理解与路由器（跨模态检索优化核心）
优化点：
  1. 视觉引用识别：自动识别"图3"、"第11页图3"、"图示"等视觉引用
  2. 返回图号、页码、部件编号等结构化元数据，供检索器做强化召回
  3. 区分纯文本问题与图文强关联问题
"""
import re
from typing import Dict, List, Optional

# 视觉引用正则
_FIG_PATTERNS = [
    r"第\s*(\d+)\s*页\s*图\s*(\d+)",      # 第11页图3
    r"图\s*(\d+)",                          # 图3
    r"第\s*(\d+)\s*页",                     # 第11页
    r"附图\s*(\d+)",                        # 附图3
]

# 部件编号提取
_PART_PATTERN = re.compile(r"(?:部件|编号)\s*(\d+)")

# 图像问题关键词（涉及图纸/部件/位置关系）
_IMAGE_HINTS = [
    r"第\s*\d+\s*页\s*图", r"图片中", r"图示", r"图中", r"图纸", r"附图",
    r"部件\s*\d", r"部件[一二三四五六七八九十]", r"编号\s*\d+",
    r"位置关系", r"位于", r"经过", r"进入", r"调整.*位置", r"哪个部件",
]


def detect_figure_refs(query: str) -> Dict:
    """识别问题中的视觉引用：图号、页码、部件编号列表"""
    fig_no = None
    page = None
    # 第11页图3
    m = re.search(r"第\s*(\d+)\s*页\s*图\s*(\d+)", query)
    if m:
        page, fig_no = int(m.group(1)), int(m.group(2))
    else:
        m2 = re.search(r"图\s*(\d+)", query)
        if m2:
            fig_no = int(m2.group(1))
        m3 = re.search(r"第\s*(\d+)\s*页", query)
        if m3:
            page = int(m3.group(1))
    parts = [int(x) for x in _PART_PATTERN.findall(query)]
    return {"fig_no": fig_no, "page": page, "parts": parts}


def is_image_query(query: str) -> bool:
    """判断是否为图文强关联问题（需要图像描述参与检索）"""
    refs = detect_figure_refs(query)
    if refs["fig_no"] is not None or refs["page"] is not None:
        return True
    for pattern in _IMAGE_HINTS:
        if re.search(pattern, query):
            return True
    return False


def rewrite_query_for_retrieval(query: str, fig_desc: Optional[str] = None) -> str:
    """
    跨模态查询重写：将视觉引用信息注入查询文本，增强检索召回。
    当检测到图号引用且有该图的结构化描述时，把描述拼入查询。
    """
    refs = detect_figure_refs(query)
    enhanced = query
    if fig_desc:
        enhanced = f"{query}\n【相关图纸描述】{fig_desc}"
    return enhanced


def route_query(query: str) -> Dict:
    """路由查询，返回跨模态检索所需的全部元数据"""
    refs = detect_figure_refs(query)
    img = is_image_query(query)
    return {
        "query_type": "image" if img else "text",
        "needs_vlm": img,
        "fig_no": refs["fig_no"],
        "page": refs["page"],
        "parts": refs["parts"],
        "has_visual_ref": refs["fig_no"] is not None or refs["page"] is not None,
    }
