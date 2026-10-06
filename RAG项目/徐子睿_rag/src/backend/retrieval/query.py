# -*- coding: utf-8 -*-
"""retrieval/query.py —— 查询规范化与规则改写。

在链路中的位置：
    retrieve_with_trace 的第一步：把用户口语问题变成"适合检索的检索串"，
    之后再交给两路召回。

两个函数：
    normalize_query  术语写法归一化 + 去掉疑问句式
    rewrite_query    按规则表补全专业术语，并返回改写前后的对照

为什么用规则而不是让大模型改写：规则快、零成本、结果可解释、可复现，
答辩时能明确指出"是哪条规则救回了这题"。
"""
from __future__ import annotations

import re
from typing import Any

from .config import REWRITE_RULES

def normalize_query(text: str) -> str:
    """把口语化问题规范化成更适合检索的形式。

    参数：
        text: 用户原始问题
    返回：
        去掉疑问词和标点、术语写法统一后的检索串。

    做两件事：
        1. 术语归一化 —— 和 pipeline.normalize_text 保持同一套规则。
           这一步是必需的：入库时把 "SF 6" 统一成了 "SF6"，如果查询不统一，
           BM25 路就永远匹配不上（同样的文本必须以同样的形式参与字面匹配）。
        2. 去掉"请问/如何/哪些"这类疑问句式 —— 它们是问句的语法骨架，
           在文档正文里不会出现，留着只会让 BM25 把它们当检索词匹配噪声。
    """
    text = (text or "").strip().replace("⁃", "-").replace("—", "-").replace("－", "-")
    text = re.sub(r"\bSF\s+6\b", "SF6", text, flags=re.I)
    text = re.sub(r"\b(GB|DL)\s*/\s*T\s*(\d+(?:\s*\.\s*\d+)?)\s*-\s*(\d{3})\s*(\d)\b", r"\1/T \2-\3\4", text, flags=re.I)
    text = re.sub(r"\b(GB|DL)\s*/\s*T\s*(\d+(?:\s*\.\s*\d+)?)\s*-\s*(\d{4})\b", r"\1/T \2-\3", text, flags=re.I)
    text = re.sub(r"请问|请|简述|介绍|说明|怎么|如何|哪些|什么|需要", " ", text)
    text = re.sub(r"[？?！!，,；;：:]+", " ", text)
    return re.sub(r"[ \t　]+", " ", text).strip()

def rewrite_query(query: str) -> dict[str, Any]:
    """做查询改写：命中规则表就补全术语，返回值同时带上改写前后对照。

    参数：
        query: 用户原始问题
    返回：
        {
          "original_query":   原始问题（原样保留，用于前端展示"你问的是…"）
          "normalized_query": 规范化后的查询
          "rewritten_query":  补全术语后的最终检索串
          "expanded_terms":   新补充进来的术语（不含原查询），前端高亮"改写了什么"
          "changed":          是否发生了改写（前端据此决定要不要显示改写这一步）
        }

    除规则表外的三条通用兜底规则：
        - 问"功能/要求/条件"且提到"装置/设备" → 补"功能""要求"（文档里的小节标题用词）
        - 提到"标准/规范/引用文件" → 补"规范性引用文件"（国标里的固定章节名）
        - 提到"回收/净化/回充" → 补"现场装置"（这几个能力都归属该章节）
    """
    normalized = normalize_query(query)
    terms = [normalized] if normalized else []
    for keyword, extras in REWRITE_RULES.items():
        if keyword in normalized:
            terms.extend(extras)
    if any(word in normalized for word in ("功能", "要求", "条件")) and any(word in normalized for word in ("装置", "设备")):
        terms.extend(["功能", "要求"])
    if any(word in normalized for word in ("标准", "规范", "引用文件")):
        terms.extend(["规范性引用文件", "标准", "条款"])
    if any(word in normalized for word in ("回收", "净化", "回充")):
        terms.append("现场装置")

    # dict.fromkeys 去重且保序：原查询词必须在最前面，扩展词顺序也保持稳定可复现
    unique = list(dict.fromkeys(term.strip() for term in terms if term.strip()))
    rewritten = " ".join(unique)
    return {
        "original_query": query,
        "normalized_query": normalized,
        "rewritten_query": rewritten or normalized,
        "expanded_terms": unique[1:],
        "changed": rewritten != normalized,
    }
