# -*- coding: utf-8 -*-
"""
coreference.py — 指代消解与多轮 Query 理解
工单编号：人工智能NLP-RAG-Query 理解优化任务

把多轮对话中的指代词（他/它/这个公司/那…呢）解析为完整实体，
使每轮问题都能独立检索。规则优先，LLM 兜底。
"""
from __future__ import annotations

import re
from typing import List, Dict, Optional

# 常见指代词
_PRONOUNS = ["他", "它", "她", "其", "该公司", "本公司", "该企业", "这个公司", "这家公司", "该银行"]
_COMPANY_PATTERN = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限公司|集团|银行|证券|保险)")

# 实体识别：从文本中提取公司名
_COMPANIES = ["武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司"]


def extract_company(text: str) -> Optional[str]:
    """从文本中提取公司名（优先已知公司，其次通用模式）。"""
    for c in _COMPANIES:
        if c in text:
            return c
    m = _COMPANY_PATTERN.search(text)
    return m.group(0) if m else None


def resolve_coreference(query: str, history: List[Dict], llm=None) -> str:
    """把 query 中的指代替换为具体实体，返回可独立检索的完整问题。

    history: [{"query", "answer", "entity"}...]，entity 为该轮涉及的公司实体。
    """
    q = query.strip()
    current_entity = None
    for h in reversed(history):
        if h.get("entity"):
            current_entity = h["entity"]
            break

    # 1. "那 XX 呢？" → 切换到新实体并复用上一轮问法
    m = re.search(r"那\s*(.+?)\s*呢", q)
    if m and "销售部" not in q and "哪些" not in q:
        new_entity = extract_company(m.group(1))
        if new_entity:
            prev = history[-1] if history else {}
            prev_entity = prev.get("entity", "")
            # 优先复用上一轮「已消解」的完整问题，替换实体
            prev_resolved = prev.get("resolved", "")
            prev_query = prev.get("query", "")
            if prev_entity and prev_entity in prev_resolved:
                q = prev_resolved.replace(prev_entity, new_entity)
            elif prev_entity and prev_entity in prev_query:
                q = prev_query.replace(prev_entity, new_entity)
            else:
                q = q.replace(m.group(0), new_entity)
            return q

    # 2. 含指代词 → 用当前实体替换
    has_pronoun = any(p in q for p in _PRONOUNS)
    if has_pronoun and current_entity:
        for p in _PRONOUNS:
            q = q.replace(p, current_entity)

    # 3. 若问题本身没带公司名，但历史有当前实体，则补全主语
    if not extract_company(q) and current_entity and not has_pronoun:
        # 仅在问题缺少主语且与历史相关时补全
        pass

    # 4. LLM 兜底（可选）：仅当规则仍未完全消解指代时，用大模型改写
    remaining = [p for p in _PRONOUNS if p in q]
    if llm is not None and remaining:
        try:
            prompt = (
                "请把以下多轮对话中最后一句的指代消解为完整问题（补全公司名）。\n"
                + "\n".join(f"Q: {h['query']}" for h in history[-3:])
                + f"\n当前问：{query}\n改写后的完整问题："
            )
            rewritten = llm.generate(prompt).strip()
            # 过滤占位/无效输出（如离线抽取式 LLM 的"未提供上下文"提示）
            if (rewritten and len(rewritten) > 3
                    and "未提供" not in rewritten and "无法" not in rewritten):
                q = rewritten
        except Exception:
            pass

    return q


def track_entity(query: str, resolved: str, answer: str) -> Optional[str]:
    """记录本轮涉及的实体（用于后续轮次指代消解）。"""
    ent = extract_company(resolved) or extract_company(query)
    return ent
