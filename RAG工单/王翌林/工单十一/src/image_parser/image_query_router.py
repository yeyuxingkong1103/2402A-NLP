# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_query_router.py —— 工单四图像感知查询路由（新增文件）

职责：判断问题是否涉及图像内容，输出路由模式与置信度，供 rag_engine_v4 分流：
  image_first —— 命中图像关键词，优先图像检索（仍回退文本/表格兜底）
  hybrid      —— 未明确命中，走工单三三路混合
"""
import re
from typing import Any, Dict

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

# 工单四：图像类问题关键词（覆盖 id5 组织结构图 / id6 增长图）
# 注：单字"图"不参与子串匹配（公司名"兴图"会误命中），由 _IMG_PAT 正则接管
IMAGE_KEYWORDS = (
    "图", "组织结构", "增长图", "图表", "示意图", "架构图", "流程图",
    "结构图", "柱状", "饼图", "折线", "曲线", "销售部", "增长率",
    "如下图", "图示", "图中", "组织架构", "chart", "diagram", "figure",
)
# 工单四：强信号词（单命中即 image_first）
_STRONG = ("组织结构", "结构图", "增长图", "流程图", "示意图", "架构图",
           "饼图", "柱状", "折线", "图中", "如下图", "组织架构")
# 工单四：单字"图"的正则形态——"XX图"（图后不接汉字，防"兴图新科"误命中）、"图中/图示"、"如图"
_IMG_PAT = re.compile(r"[\u4e00-\u9fa5]{2,}图(?![\u4e00-\u9fa5])|图[中示内]|如图")


def route_query(query: str) -> Dict[str, Any]:
    """工单四：路由判断 → {"mode", "confidence", "hits"}"""
    hits = sorted({kw for kw in IMAGE_KEYWORDS
                   if len(kw) > 1 and kw.lower() in query.lower()})
    if _IMG_PAT.search(query):                       # 工单四：单字"图"按正则识别
        hits.append("图")
        hits = sorted(set(hits))
    strong = [h for h in hits if h in _STRONG]
    if strong:                                       # 工单四：强信号直接 image_first
        return {"mode": "image_first", "confidence": min(0.95, 0.7 + 0.08 * len(strong)),
                "hits": hits}
    if hits:                                         # 工单四：弱信号（如"销售部"）降权
        return {"mode": "image_first", "confidence": min(0.6, 0.35 + 0.1 * len(hits)),
                "hits": hits}
    return {"mode": "hybrid", "confidence": 0.5, "hits": []}  # 工单四：默认混合
