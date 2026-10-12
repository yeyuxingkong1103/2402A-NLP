# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
Query 理解模块：实现意图识别、消歧（指代消解）、分解与抽象，
支撑多轮对话中的指代与话题继承。
"""
import re

# 实体别名表（规范名 -> 别名列表）
ENTITIES = [
    ("武汉兴图新科电子股份有限公司", ["武汉兴图新科电子股份有限公司", "兴图新科", "兴图新科电子", "他", "它", "这个公司", "该公司", "该企业"]),
    ("武汉力源信息技术股份有限公司", ["武汉力源信息技术股份有限公司", "力源信息", "力源"]),
]

# 意图关键词（用于识别问题意图与话题继承）
INTENT_PATTERNS = [
    ("军用领域收入", ["军用", "收入"]),
    ("技术标准", ["技术标准", "标准"]),
    ("主营业务占比", ["比重", "占比", "主营业务"]),
    ("上游企业", ["上游"]),
    ("下游行业", ["下游"]),
    ("供应商", ["供应商"]),
    ("国家科技进步一等奖", ["科技进步", "一等奖", "工程"]),
    ("注册资本", ["注册资本"]),
    ("法定代表人", ["法定代表人", "法人"]),
    ("补充流动资金", ["补充流动资金", "流动资金"]),
    ("发行股数", ["发行股数", "发行", "股本", "总股本"]),
    ("募集资金项目", ["募集资金", "投资项目"]),
    ("关联方", ["关联方", "控制关系"]),
    ("组织结构", ["组织结构", "销售部", "销售处"]),
    ("市场增长", ["IC", "市场", "增长率", "增长"]),
]


def find_entity(text):
    """识别问题中提及的公司实体，返回规范名。"""
    for canon, aliases in ENTITIES:
        for a in aliases:
            if a in text:
                return canon
    return None


def strip_entity(text):
    """去掉实体，得到问题意图模板（用于话题继承）。"""
    for _, aliases in ENTITIES:
        for a in aliases:
            text = text.replace(a, "")
    return re.sub(r"[，。？！呢、\s]+$", "", text).strip("的")


def recognize_intent(text):
    """意图识别：返回命中的意图标签。"""
    for label, kws in INTENT_PATTERNS:
        if any(k in text for k in kws):
            return label
    return "通用问题"


def resolve_coreference(query, history):
    """
    指代消歧 + 话题继承：
    1. 代词/指代（他/这个公司/该公司）→ 上一轮实体；
    2. “那X呢？”→ 切换实体 X，继承上一问意图。
    history: [(question, answer)]，按时间顺序。
    """
    if not history:
        return query

    # 1) 代词与指代消解
    has_pronoun = any(k in query for k in ["他", "它", "这个公司", "该公司", "该企业"])
    if has_pronoun:
        entity = _last_entity(history)
        if entity:
            for k in ["这个公司", "该公司", "该企业", "他", "它"]:
                query = query.replace(k, entity)

    # 2) “那X呢？”话题继承
    m = re.match(r"^那(.+?)呢[？?]?$", query.strip())
    if m:
        new_entity_name = m.group(1)
        prev_intent = strip_entity(history[-1][0])
        query = f"{new_entity_name}{prev_intent}？"

    return query


def _last_entity(history):
    """从历史记录中返回最近一次出现的实体。"""
    for q, _ in reversed(history):
        e = find_entity(q)
        if e:
            return e
    return None


def decompose(query):
    """分解与抽象：把复杂问题拆成若干子问题（简单实现）。"""
    subs = []
    if "分别" in query or "多少" in query and ("和" in query or "、" in query):
        subs.append(query)
    return subs or [query]


def understand(query, history):
    """
    Query 理解总入口：返回 (改写后的问题, 意图, 实体)。
    """
    resolved = resolve_coreference(query, history)
    intent = recognize_intent(resolved)
    entity = find_entity(resolved)
    return resolved, intent, entity
