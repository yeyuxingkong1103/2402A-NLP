# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：多文档路由（工单3 新增）

工单要求「问题里出现公司名时，能自动路由到对应文档（如"武汉力源"→招股说明书2，
"武汉兴图新科"→招股说明书1）」。

做法（规则版，<1ms，无需模型）：
  1. 别名表匹配问题文本 → 得到目标 source 列表；
  2. 命中唯一 source → 检索时用 Qdrant payload filter 限定 `source`；
  3. 未命中（或同时命中两家公司，如对比类问题）→ 不加过滤，全库检索；
  4. `fallback_open`：过滤后候选不足时放开过滤重检（路由误判不至于空召回）。

为什么用规则而非模型：
  公司名是**封闭集合**（本工单只有两家），规则准确率 100% 且零延迟；
  工单同时要求响应 ≤3 秒，把预算留给检索与生成更划算。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import re

from src import config

# 页码范围：招股说明书里的"报告期/最近三年"等相对表述无需路由处理

NOISE_PREFIXES = ("根据", "依据", "参见", "按照")


def _normalize(text: str) -> str:
    """去空白与全角空格，便于别名匹配（"武汉 力源" 也能命中）。"""
    return re.sub(r"[\s　]+", "", text or "")


def match_sources(question: str) -> list[str]:
    """问题 → 命中的 source 列表（保持 config.SOURCE_NAMES 的顺序）。"""
    q = _normalize(question)
    if not q:
        return []
    hits: list[str] = []
    for source in config.SOURCE_NAMES:
        aliases = config.DOC_ALIASES.get(source, ())
        if any(_normalize(a) in q for a in aliases):
            hits.append(source)
    return hits


def route(question: str) -> list[str] | None:
    """路由结果：唯一命中返回 [source]；未命中或多命中返回 None（=不过滤）。"""
    if not config.DOC_ROUTER_ENABLED:
        return None
    hits = match_sources(question)
    return hits if len(hits) == 1 else None


def build_filter(sources: list[str] | None,
                 exclude_block_types: list[str] | None = None):
    """source 列表 + 需排除的块类型 → Qdrant Filter（都为空 → 不加过滤）。

    exclude_block_types 供「无表格优化」消融对比使用：把 table_row / table_header
    排除掉，等价于"表格只以朴素整块文本存在"，用来量化表格结构化的净增益。
    """
    if not sources and not exclude_block_types:
        return None
    try:
        from qdrant_client.http import models as qm

        must, must_not = [], []
        if sources:
            must.append(qm.FieldCondition(
                key="source", match=qm.MatchAny(any=list(sources))))
        if exclude_block_types:
            must_not.append(qm.FieldCondition(
                key="block_type", match=qm.MatchAny(any=list(exclude_block_types))))
        return qm.Filter(must=must or None, must_not=must_not or None)
    except Exception:  # noqa: BLE001 —— 过滤构造失败时退化为不过滤
        return None


def describe(question: str) -> dict:
    """路由详情（界面展示与调试用）。"""
    hits = match_sources(question)
    routed = route(question)
    if routed:
        reason = f"命中公司别名 → 限定《{routed[0]}》"
    elif len(hits) > 1:
        reason = f"同时命中 {len(hits)} 家公司 → 全库检索"
    else:
        reason = "未命中公司名 → 全库检索"
    return {"sources": routed, "matched": hits, "reason": reason}
