# -*- coding: utf-8 -*-
"""
实体+关系抽取器 V2 - 扩展词典 + 同义词 + LLM增强
工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务
"""
import re, json, os, logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# ============ 扩展词典 (V8 升级) ============
COMPANY_PATTERNS = [
    r"武汉兴图新科电子股份有限公司",
    r"武汉兴图新科",
    r"武汉力源信息技术股份有限公司",
    r"武汉力源",
    r"武汉力源科技有限公司",
    r"武汉力源科技",
    r"发行人", r"本公司",
]

COMPANY_SYNONYMS = {
    "武汉兴图新科电子股份有限公司": ["武汉兴图新科", "兴图新科", "武汉兴图"],
    "武汉力源信息技术股份有限公司": ["武汉力源", "力源信息", "武汉力源股份"],
    "武汉力源科技有限公司": ["武汉力源科技", "力源科技"],
}

FINANCIAL_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*万元",
    r"(\d+(?:\.\d+)?)\s*亿元",
    r"(\d+(?:\.\d+)?)\s*万股",
    r"(\d+(?:\.\d+)?)\s*%",
    r"报告期内", r"注册资本", r"发行股数", r"募集资金",
]

PERSON_PATTERNS = [
    r"法定代表人[是为](.{1,10})",
    r"董事长[是为](.{1,10})",
    r"总经理[是为](.{1,10})",
    r"实际控制人[是为](.{1,10})",
]

STANDARD_PATTERNS = [
    r"AVS\s*编解码?技术标准",
    r"国家科技进步一等奖",
    r"国家科技进步奖",
    r"技术标准",
    r"参与制定",
]

INDUSTRY_PATTERNS = [
    r"军用领域", r"电子信息行业", r"汽车电子",
    r"消费电子", r"通讯设备", r"嵌入式系统",
]

# ============ V9 新增: 关系模式扩展 ============
RELATION_PATTERNS = [
    # (源模式, 关系类型, 目标模式)
    (r"(.{2,20}公司).{0,10}?法定代表人[是为](.{1,10})", "法定代表人"),
    (r"(.{2,20}公司).{0,10}?参与制定了?(.{2,20}?标准)", "参与制定"),
    (r"(.{2,20}公司).{0,10}?注册资本[为是](.{5,15})", "注册资本"),
    (r"(.{2,20}公司).{0,10}?来自(.{2,20}?领域).{0,10}?收入", "来自"),
    (r"(.{2,20}公司).{0,10}?(.{2,15}?科技).{0,5}?控股股东", "控股股东"),
    (r"(.{2,20}科技).{0,5}?控股股东.{0,5}?(.{2,20}公司)", "控股股东_of"),
    (r"(.{2,20}部门).{0,5}?下属.{0,5}?(.{2,20}部门)", "下属"),
    (r"(.{2,20}部门).{0,5}?属于.{0,5}?(.{2,20}部门)", "属于"),
    (r"(.{2,20}公司).{0,5}?募集资金.{0,5}?(.{2,20}?项目)", "募集资金投资"),
    (r"(.{2,20}公司).{0,5}?发行.{0,5}?(.{0,15}?万股)", "本次发行"),
    (r"(.{2,20}行业).{0,5}?增长率.{0,5}?(.{0,10})", "增长率"),
    (r"(.{2,20}行业).{0,5}?负增长", "负增长"),
]


def extract_entities(text: str) -> List[Dict]:
    """V2 实体抽取 - 扩展模式"""
    entities = []
    seen = set()

    def add_ent(eid: str, etype: str):
        key = f"{eid}|{etype}"
        if key not in seen:
            entities.append({"id": eid, "type": etype})
            seen.add(key)

    # 公司
    for pat in COMPANY_PATTERNS:
        for m in re.finditer(pat, text):
            add_ent(m.group(), "Company")

    # 金融
    for pat in FINANCIAL_PATTERNS:
        for m in re.finditer(pat, text):
            if m.groups():
                add_ent(m.group(), "Financial")

    # 人物
    for pat in PERSON_PATTERNS:
        for m in re.finditer(pat, text):
            name = m.group(1).strip()
            if name and len(name) >= 2:
                add_ent(name, "Person")

    # 标准
    for pat in STANDARD_PATTERNS:
        for m in re.finditer(pat, text):
            add_ent(m.group(), "Standard")

    # 行业
    for pat in INDUSTRY_PATTERNS:
        for m in re.finditer(pat, text):
            add_ent(m.group(), "Industry")

    return entities


def extract_relations(text: str, entities: List[Dict] = None) -> List[Dict]:
    """V2 关系抽取 - 扩展模式"""
    relations = []
    for src_pat, rel_type in RELATION_PATTERNS:
        m = re.search(src_pat, text)
        if m and len(m.groups()) >= 2:
            src = m.group(1).strip()
            tgt = m.group(2).strip()
            # 跳过过短/过长
            if 2 <= len(src) <= 30 and len(tgt) >= 2:
                relations.append({
                    "source": src, "target": tgt,
                    "relation": rel_type, "weight": 0.85,
                })
    return relations


def entity_disambiguation(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    实体消歧: 同名实体根据上下文区分

    例: "控股股东" → 武汉力源科技(不是武汉力源)
    """
    disambiguated = []
    for c in candidates:
        nid = c["node"]
        score = c.get("score", 0)

        # 上下文匹配
        if "控股股东" in query and "科技" in nid:
            score += 0.3
        if query in nid or nid in query:
            score += 0.2
        # 同义词匹配
        for canonical, syns in COMPANY_SYNONYMS.items():
            if nid in syns or canonical in nid:
                if canonical in query or any(s in query for s in syns):
                    score += 0.15

        c["score"] = min(score, 1.0)
        disambiguated.append(c)

    disambiguated.sort(key=lambda x: x["score"], reverse=True)
    return disambiguated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    text = "武汉力源科技有限公司是武汉力源信息技术股份有限公司的控股股东(持股35%)。"
    ents = extract_entities(text)
    rels = extract_relations(text, ents)
    print(f"实体: {len(ents)}")
    for e in ents: print(f"  {e}")
    print(f"关系: {len(rels)}")
    for r in rels: print(f"  {r}")
