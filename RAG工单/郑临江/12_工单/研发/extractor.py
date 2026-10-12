# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化
实体与关系抽取：根据 PDF 内容优化实体类型、关系类型。
规则抽取为主，配置 LLM API Key 时切换为 LLM 抽取。
"""
import json
import re

import config
from llm import call_llm

COMPANY_RE = re.compile(r"[一-龥]{2,30}?(?:股份有限公司|有限责任公司|有限公司|集团|研究院|公司)")
MONEY_RE = re.compile(r"[\d,，.]+\s*(?:万元|亿元|元|美元|人民币)")
PERCENT_RE = re.compile(r"[\d.]+%")
DATE_RE = re.compile(r"\d{4}年(?:\d{1,2}月)?")
PERSON_HINTS = ["法定代表人", "董事长", "总经理", "实际控制人", "财务负责人", "董事", "监事"]

# 关系类型 -> 触发关键词
RELATION_RULES = [
    ("持股", ["持股", "持有", "出资", "股权"]),
    ("控股", ["控股", "控制权", "实际控制"]),
    ("发行", ["发行股数", "公开发行", "发行后总股本"]),
    ("投资", ["投资", "募集资金"]),
    ("参与制定", ["制定", "参与制定", "技术标准"]),
    ("主营", ["主营业务", "主营"]),
    ("关联方", ["关联方", "关联关系"]),
]


def extract_entities(text):
    """返回 [(实体名, 实体类型)]，实体类型已按招股书内容优化。"""
    entities = []
    for m in COMPANY_RE.finditer(text):
        entities.append((m.group(), "公司"))
    for m in MONEY_RE.finditer(text):
        entities.append((m.group(), "金额"))
    for m in PERCENT_RE.finditer(text):
        entities.append((m.group(), "比例"))
    for m in DATE_RE.finditer(text):
        entities.append((m.group(), "日期"))
    for hint in PERSON_HINTS:
        for m in re.finditer(hint + r"[:：]?\s*([一-龥]{2,4})", text):
            entities.append((m.group(1), "人物"))
    for m in re.finditer(r"([一-龥A-Za-z0-9]{2,20}?(?:标准|规范|规程))", text):
        entities.append((m.group(1), "技术标准"))
    for m in re.finditer(r"([一-龥]{2,12}?(?:行业|产业|领域))", text):
        entities.append((m.group(1), "行业"))
    for m in re.finditer(r"([一-龥]{2,20}?(?:项目|工程|系统|平台|终端))", text):
        entities.append((m.group(1), "产品项目"))
    return entities


def extract_relations(entities, text):
    """返回 [(头实体, 关系类型, 尾实体)]，关系类型已优化。"""
    relations = []
    companies = [e[0] for e in entities if e[1] == "公司"]
    for rel, kws in RELATION_RULES:
        if any(k in text for k in kws) and len(companies) >= 2:
            relations.append((companies[0], rel, companies[1]))
    return relations


def extract_with_llm(text):
    """LLM 抽取（配置 API Key 时），限定实体/关系类型。"""
    prompt = (
        f"请从以下文本中抽取实体和关系。\n"
        f"实体类型限定为：{'、'.join(config.ENTITY_TYPES)}。\n"
        f"关系类型限定为：{'、'.join(config.RELATION_TYPES)}。\n"
        f'输出 JSON：{{"entities": [["实体名","类型"]], "relations": [["头","关系","尾"]]}}\n\n'
        f"文本：{text[:2000]}"
    )
    raw = call_llm(prompt)
    if not raw:
        return {"entities": [], "relations": []}
    try:
        return json.loads(raw)
    except Exception:
        return {"entities": [], "relations": []}
