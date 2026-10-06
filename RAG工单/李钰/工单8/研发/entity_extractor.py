# -*- coding: utf-8 -*-
"""
实体+关系抽取器 (轻量规则版)
工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答

进阶: 可用 spaCy / LLM 替换规则版
"""
import re
import json
import os
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


# 实体词典 (金融领域)
COMPANY_PATTERNS = [
    r"武汉兴图新科电子股份有限公司",
    r"武汉兴图新科",
    r"武汉力源信息技术股份有限公司",
    r"武汉力源",
    r"武汉力源科技有限公司",
    r"武汉力源科技",
]

FINANCIAL_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*万元",
    r"(\d+(?:\.\d+)?)\s*亿元",
    r"(\d+(?:\.\d+)?)\s*万股",
    r"(\d+(?:\.\d+)?)\s*%",
]

STANDARD_PATTERNS = [
    r"AVS\s*编解码?技术标准",
    r"国家科技进步一等奖",
    r"国家科技进步奖",
]

INDUSTRY_PATTERNS = [
    r"军用领域",
    r"电子信息行业",
    r"汽车电子",
    r"消费电子",
]


def extract_entities(text: str) -> List[Dict]:
    """从文本中抽取实体"""
    entities = []
    seen = set()

    # 公司
    for pat in COMPANY_PATTERNS:
        for m in re.finditer(pat, text):
            eid = f"Company_{m.group()}"
            if eid not in seen:
                entities.append({"id": m.group(), "type": "Company"})
                seen.add(eid)

    # 金融数据
    for pat in FINANCIAL_PATTERNS:
        for m in re.finditer(pat, text):
            val = m.group()
            eid = f"Financial_{val}"
            if eid not in seen:
                entities.append({"id": val, "type": "Financial"})
                seen.add(eid)

    # 标准/奖项
    for pat in STANDARD_PATTERNS:
        for m in re.finditer(pat, text):
            eid = f"Standard_{m.group()}"
            if eid not in seen:
                entities.append({"id": m.group(), "type": "Standard"})
                seen.add(eid)

    # 行业
    for pat in INDUSTRY_PATTERNS:
        for m in re.finditer(pat, text):
            eid = f"Industry_{m.group()}"
            if eid not in seen:
                entities.append({"id": m.group(), "type": "Industry"})
                seen.add(eid)

    logger.info(f"[实体抽取] {len(entities)} 个实体")
    return entities


def extract_relations(text: str, entities: List[Dict]) -> List[Dict]:
    """
    从文本中抽取关系 (规则版)

    支持的关系模式:
      - (A) 是 (B) 的法定代表人
      - (A) 持股 (B) %
      - (A) 参与制定 (B)
      - (A) 的注册资本是 (B)
      - (A) 来自 (B)
    """
    relations = []

    # 关系模式: (公司, 参与制定, 标准)
    for std in re.finditer(r"参与制定了?(.{1,20}?标准)", text):
        target = std.group(1)
        for ent in entities:
            if ent["type"] == "Company" and ent["id"] in text[max(0, std.start()-30):std.start()]:
                relations.append({
                    "source": ent["id"],
                    "target": target,
                    "relation": "参与制定",
                })
                break

    # 关系模式: (公司, 法定代表人, 人物)
    for m in re.finditer(r"法定代表人[是为](.{1,10})", text):
        target = m.group(1).strip()
        for ent in entities:
            if ent["type"] == "Company" and ent["id"] in text[max(0, m.start()-30):m.start()]:
                relations.append({
                    "source": ent["id"],
                    "target": target,
                    "relation": "法定代表人",
                })
                break

    # 关系模式: (公司, 注册资本, 金额)
    for m in re.finditer(r"注册资本[为是](.{5,15})", text):
        target = m.group(1).strip()
        for ent in entities:
            if ent["type"] == "Company" and ent["id"] in text[max(0, m.start()-30):m.start()]:
                relations.append({
                    "source": ent["id"],
                    "target": target,
                    "relation": "注册资本",
                })
                break

    # 关系模式: (公司, 获科技进步奖)
    for m in re.finditer(r"参与.{0,5}?工程.{0,10}?(获|荣获)(.{5,10}?一等奖)", text):
        for ent in entities:
            if ent["type"] == "Company":
                relations.append({
                    "source": ent["id"],
                    "target": m.group(2),
                    "relation": "参与的工程获",
                })
                break

    logger.info(f"[关系抽取] {len(relations)} 条关系")
    return relations


def build_kg_from_texts(chunks: List[Dict]) -> Dict:
    """从文本块构建知识图谱"""
    all_entities = []
    all_relations = []
    seen_ent = set()
    seen_rel = set()

    for chunk in chunks:
        text = chunk.get("text", "")
        entities = extract_entities(text)
        relations = extract_relations(text, entities)

        for e in entities:
            eid = f"{e['id']}|{e['type']}"
            if eid not in seen_ent:
                all_entities.append(e)
                seen_ent.add(eid)

        for r in relations:
            rid = f"{r['source']}|{r['relation']}|{r['target']}"
            if rid not in seen_rel:
                all_relations.append({**r, "weight": 0.8})
                seen_rel.add(rid)

    return {"nodes": all_entities, "edges": all_relations}


def load_preset_kg(path: str) -> Dict:
    """加载预设知识图谱"""
    if not os.path.exists(path):
        return {"nodes": [], "edges": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 演示
    text = "武汉兴图新科电子股份有限公司的注册资本为7360万元,参与制定了AVS编解码技术标准。"
    ents = extract_entities(text)
    rels = extract_relations(text, ents)
    print(f"实体: {ents}")
    print(f"关系: {rels}")
