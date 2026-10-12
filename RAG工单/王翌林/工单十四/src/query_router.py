# -*- coding: utf-8 -*-
"""
工单十四：查询路由器
区分纯文本问题和图像问题（涉及图纸/图片/部件/尺寸的走多模态VQA通道）
"""
import re
from typing import Dict

# 图像问题关键词：涉及图纸、图片、部件位置、尺寸标注、气流方向等
_IMAGE_HINTS = [
    r"第\s*\d+\s*页\s*图", r"图片中", r"图示", r"图中", r"图纸",
    r"部件\s*\d", r"部件[一二三四五六七八九十]",
    r"位置关系", r"间隔距离", r"气流方向", r"首先经过", r"紧接着",
    r"尺寸\s*[xX]\d", r"h\d", r"外壳直径",
]


def is_image_query(query: str) -> bool:
    """判断是否为图像相关问题（需要多模态VQA）"""
    for pattern in _IMAGE_HINTS:
        if re.search(pattern, query):
            return True
    return False


def detect_page(query: str) -> int:
    """从问题中提取页码（如"第7页图片"）"""
    m = re.search(r"第\s*(\d+)\s*页", query)
    return int(m.group(1)) if m else None


def route_query(query: str) -> Dict:
    """路由查询：返回路由类型和相关元数据"""
    img = is_image_query(query)
    page = detect_page(query)
    return {
        "query_type": "image" if img else "text",
        "page": page,
        "needs_vlm": img,
    }
