# -*- coding: utf-8 -*-
"""多轮 Query 改写（本工单核心新增）。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

把带上下文依赖的「用户口语问句」改写为「可独立检索的完整问句」，解决两类问题：

  1) 指代消解（Coreference Resolution）
       Q2「他参与的哪个工程荣获了国家科技进步一等奖？」——「他」指代上一轮的兴图新科；
       Q3「这个公司的法定代表人是谁？」——「这个公司」同样指代兴图新科。

  2) 省略补全（Ellipsis Resolution）
       Q4「那武汉力源信息技术股份有限公司呢？」——省略了谓语（法定代表人是谁），
       需继承上一轮意图骨架，替换主语公司。

改写结果同时给出文档路由提示（doc_hint）与继承的实体，供检索阶段使用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src import config
from src.conversation import Conversation
from src.query_understanding import matched_alias, strip_noise

_ALL_ALIASES = [a for aliases in config.DOC_COMPANY.values() for a in aliases]

# 指代词（长词优先，避免「这个公司」被「这」抢先匹配）
_COREF = sorted(set(config.COREF_PRONOUNS), key=len, reverse=True)


@dataclass
class RewriteResult:
    """改写结果。"""
    original: str                      # 用户原始问句
    rewritten: str                     # 改写后的独立问句
    need_rewrite: bool = False         # 是否发生了改写
    coref: bool = False                # 是否命中指代消解
    ellipsis: bool = False             # 是否命中省略补全
    resolved_entity: str = ""          # 指代/省略补全后的主语实体
    inherited_intent: str = ""         # 继承的意图骨架
    doc_hint: str = ""                 # 文档路由提示（源文件名）
    reasons: List[str] = field(default_factory=list)


def extract_intent(question: str) -> str:
    """从问句中剥离公司名 / 指代词 / 套话，得到意图骨架。

    例：「这个公司的法定代表人是谁？」→「法定代表人是谁」
        「报告期内，武汉兴图新科…来自军用领域的收入分别是多少？」→「来自军用领域的收入」
    """
    t = question or ""
    for alias in sorted(_ALL_ALIASES, key=len, reverse=True):
        t = t.replace(alias, "")
    for p in _COREF:
        t = t.replace(p, "")
    t = strip_noise(t)
    t = re.sub(r"^(来自|来自|其|该|上述)?\s*", "", t)
    return t.strip(" 的了")


def _has_coref(question: str) -> str:
    """返回命中的指代词（未命中返回空串）。"""
    for p in _COREF:
        if p in (question or ""):
            return p
    return ""


def _is_ellipsis(question: str) -> bool:
    q = (question or "").strip()
    for pat in config.ELLIPSIS_PATTERNS:
        if re.search(pat, q):
            return True
    return False


class QueryRewriter:
    """基于规则的多轮 Query 改写器（确定性、毫秒级）。"""

    def rewrite(self, question: str, conv: Conversation) -> RewriteResult:
        q = (question or "").strip()
        res = RewriteResult(original=q, rewritten=q)
        if not q:
            return res

        explicit_src, explicit_alias = matched_alias(q)
        coref_word = _has_coref(q)
        ellipsis = _is_ellipsis(q)
        active_company = conv.active_company
        active_doc = conv.active_doc
        last_intent = conv.last.intent if conv.last else ""

        # ---- 场景 A：省略补全（那 X 呢？） ---------------------------------
        if ellipsis:
            res.ellipsis = True
            res.need_rewrite = True
            target = explicit_alias or active_company
            res.resolved_entity = target
            res.inherited_intent = last_intent
            if last_intent:
                res.rewritten = f"{target}{last_intent}" if target else last_intent
                res.reasons.append(f"省略补全：继承上一轮意图「{last_intent}」")
            else:
                res.rewritten = q
                res.reasons.append("省略补全：无历史意图可继承，保持原问句")
            if target:
                res.reasons.append(f"主语补全为「{target}」")
            res.doc_hint = matched_alias(res.rewritten)[0] or explicit_src or active_doc
            return res

        # ---- 场景 B：指代消解（他 / 这个公司 …） ---------------------------
        if coref_word and active_company:
            res.coref = True
            res.need_rewrite = True
            res.resolved_entity = active_company
            rewritten = q
            # 指代词整体替换为话题公司全名（用最长别名，保证实体完整）
            full = config.DOC_COMPANY.get(active_doc, [active_company])
            full_name = max(full, key=len) if full else active_company
            rewritten = rewritten.replace(coref_word, full_name)
            res.rewritten = rewritten
            res.reasons.append(f"指代消解：「{coref_word}」→「{full_name}」")
            res.doc_hint = matched_alias(rewritten)[0] or active_doc
            return res

        # ---- 场景 C：无显式公司但属追问 → 注入话题公司 ----------------------
        if not explicit_alias and active_company and last_intent and _looks_followup(q):
            res.need_rewrite = True
            res.resolved_entity = active_company
            res.inherited_intent = last_intent
            res.rewritten = f"{active_company}{strip_noise(q)}"
            res.reasons.append(f"追问补全：注入话题公司「{active_company}」")
            res.doc_hint = matched_alias(res.rewritten)[0] or active_doc
            return res

        # ---- 场景 D：独立问句，无需改写 ------------------------------------
        res.doc_hint = explicit_src or active_doc
        return res


def _looks_followup(q: str) -> bool:
    """判断问句是否为「缺少主语」的追问（如「法定代表人是谁？」）。"""
    if len(q) > 30:
        return False
    return bool(re.search(r"(是谁|是多少|是什么|有哪些|包括哪些|怎么样|如何)", q))


def make_intent(question: str) -> str:
    """对外暴露的意图抽取（供 qa_engine 记录 Turn.intent）。"""
    return extract_intent(question)