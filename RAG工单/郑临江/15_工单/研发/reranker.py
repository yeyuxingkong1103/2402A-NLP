# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程工单
跨模态重排（Re-Ranker）：对初步检索出的图文块精细化排序，
优先选出最匹配图文语义的片段（如包含“图3”的图文块）。
"""
import re

import config
from query_understanding import detect_visual_reference


def rerank(blocks, query):
    if not config.RERANK_ENABLE:
        return blocks
    ref = detect_visual_reference(query)
    def score(b):
        s = 0.0
        if ref and ref.get("figure"):
            if ref["figure"] in b.get("figures", []):
                s += 100          # 命中目标图纸
            if b.get("is_image"):
                s += 30           # 图文块加分
        if ref and ref.get("part") is not None:
            # 部件编号与图块文本/描述的匹配密度
            part = str(ref["part"])
            s += min(b.get("text", "").count(part), 5) * 10
            s += b.get("image_desc", "").count("部件") * 5
        return s

    return sorted(blocks, key=score, reverse=True)
