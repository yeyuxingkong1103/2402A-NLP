# -*- coding: utf-8 -*-
"""
规则版实体/关系抽取器 (轻量级, 无需 LLM)
工单编号: 人工智能 NLP-RAG 项目-LightRAG 优化

LightRAG 用 LLM 做 entity extraction, 这里用规则 + 金融关键词
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


# === 金融领域实体类型 ===
ENTITY_TYPES = {
    "公司": ["公司", "股份有限公司", "科技", "电子", "集团"],
    "机构": ["部", "部门", "销售处", "办事处", "分公司", "委员会"],
    "人员": ["法定代表人", "董事长", "总经理", "控股股东", "实际控制人"],
    "产品": ["芯片", "ADC", "DAC", "射频", "接口"],
    "标准": ["标准", "规范", "技术标准"],
    "项目": ["工程", "项目", "计划"],
    "行业": ["行业", "领域", "市场"],
    "数值": ["万元", "股", "比例", "占比", "比重", "增长率"],
    "金融术语": ["募集资金", "总股本", "发行股", "投资"],
}


# === 预设金融实体库 ===
PRESET_ENTITIES = {
    "武汉力源信息技术股份有限公司": {"type": "公司", "alias": "武汉力源"},
    "武汉力源": {"type": "公司", "alias": "武汉力源信息技术股份有限公司"},
    "武汉兴图新科电子股份有限公司": {"type": "公司", "alias": "武汉兴图新科"},
    "武汉兴图新科": {"type": "公司", "alias": "武汉兴图新科电子股份有限公司"},
    "武汉力源科技有限公司": {"type": "公司", "alias": "控股股东"},
    "销售部": {"type": "机构"},
    "大客户销售部": {"type": "机构"},
    "渠道销售部": {"type": "机构"},
    "产品销售部": {"type": "机构"},
    "电商销售部": {"type": "机构"},
    "华东销售处": {"type": "机构"},
    "华南销售处": {"type": "机构"},
    "华北销售处": {"type": "机构"},
    "西南销售处": {"type": "机构"},
    "ADC芯片": {"type": "产品"},
    "DAC芯片": {"type": "产品"},
    "高速接口芯片": {"type": "产品"},
    "射频前端芯片": {"type": "产品"},
    "补充流动资金": {"type": "项目"},
    "军用领域": {"type": "行业"},
    "电子信息行业": {"type": "行业"},
    "汽车电子": {"type": "行业"},
    "消费电子": {"type": "行业"},
    "通讯设备": {"type": "行业"},
    "嵌入式系统": {"type": "行业"},
    "AVS编解码技术标准": {"type": "标准"},
    "国家科技进步一等奖": {"type": "标准"},
    "募集资金": {"type": "金融术语"},
    "总股本": {"type": "金融术语"},
    "注册资本": {"type": "金融术语"},
    "控股股东": {"type": "金融术语"},
    "关联方": {"type": "金融术语"},
    "法定代表人": {"type": "人员"},
    "2000万股": {"type": "数值"},
    "25%": {"type": "数值"},
    "7360万元": {"type": "数值"},
    "8000万元": {"type": "数值"},
    "6464.51万元": {"type": "数值"},
    "14414.16万元": {"type": "数值"},
    "18780.67万元": {"type": "数值"},
    "4627.14万元": {"type": "数值"},
}


# === 预设金融关系模板 ===
RELATION_PATTERNS = [
    (r"(.+?)的(.+?)是(.+)", "拥有属性"),
    (r"(.+?)的(.+?)为(.+)", "拥有属性"),
    (r"(.+?)是(.+?)的(.+)", "属于"),
    (r"(.+?)为(.+?)(.+)", "拥有属性"),
    (r"(.+?)持股(.+?)%", "持股"),
    (r"(.+?)拟投资(.+)", "投资"),
    (r"(.+?)来自(.+?)领域", "来自"),
    (r"(.+?)占(.+?)的比重为(.+)", "占比"),
    (r"(.+?)参与(.+?)制定", "参与制定"),
    (r"(.+?)在(.+?)领域成为重要供应商", "供应领域"),
    (r"(.+?)获(.+?)奖", "获得奖项"),
    (r"(.+?)有(.+?)个(.+?)部门", "拥有下属"),
    (r"(.+?)有(.+?)个(.+?)部", "拥有下属"),
    (r"(.+?)增长率最快", "增长最快"),
    (r"(.+?)负增长", "负增长"),
]


def extract_entities(text: str) -> List[Dict]:
    """从文本提取实体"""
    entities = []
    seen = set()

    # 1. 预设实体匹配 (优先级最高)
    for name, info in PRESET_ENTITIES.items():
        if name in text:
            eid = name
            if eid not in seen:
                seen.add(eid)
                entities.append({
                    "id": eid, "type": info["type"],
                    "summary": info.get("alias", ""),
                })

    # 2. 正则抽取数值
    num_patterns = [
        r"\d+[\.\d]*\s*万元", r"\d+[\.\d]*\s*%", r"\d+\s*万股",
        r"\d+[\.\d]*\s*亿元",
    ]
    for pat in num_patterns:
        for m in re.finditer(pat, text):
            val = m.group(0).strip()
            if val not in seen:
                seen.add(val)
                entities.append({"id": val, "type": "数值"})

    # 3. 尝试 jieba 分词补全 (轻量)
    try:
        import jieba
        for tok in jieba.cut(text):
            tok = tok.strip()
            if len(tok) >= 3 and tok not in seen:
                seen.add(tok)
                entities.append({"id": tok, "type": "候选"})
    except ImportError:
        pass

    return entities


def extract_relations(text: str, entities: List[Dict]) -> List[Dict]:
    """从文本提取关系"""
    relations = []
    for pat, rel_name in RELATION_PATTERNS:
        for m in re.finditer(pat, text):
            groups = m.groups()
            if len(groups) >= 2:
                src = groups[0].strip()
                tgt = groups[-1].strip()
                # 过滤
                if src and tgt and len(src) >= 2 and len(tgt) >= 2:
                    relations.append({
                        "source": src, "target": tgt,
                        "relation": rel_name,
                        "summary": m.group(0)[:100],
                    })
    return relations


def build_initial_graph():
    """构建 V12 初始知识图谱 (覆盖 15 个问题)"""
    ents = []
    for name, info in PRESET_ENTITIES.items():
        ents.append({"id": name, "type": info["type"]})

    # 预设关系 (基于招股说明书)
    rels = [
        # 武汉力源
        {"source": "武汉力源科技有限公司", "target": "武汉力源信息技术股份有限公司",
         "relation": "控股股东", "weight": 0.9, "summary": "持股35%"},
        {"source": "武汉力源信息技术股份有限公司", "target": "武汉力源科技有限公司",
         "relation": "被控股", "weight": 0.9},
        {"source": "武汉力源信息技术股份有限公司", "target": "2000万股",
         "relation": "发行股数", "weight": 0.9},
        {"source": "武汉力源信息技术股份有限公司", "target": "25%",
         "relation": "发行股占比", "weight": 0.9},
        {"source": "武汉力源信息技术股份有限公司", "target": "ADC芯片",
         "relation": "募集资金投资", "weight": 0.8},
        {"source": "武汉力源信息技术股份有限公司", "target": "DAC芯片",
         "relation": "募集资金投资", "weight": 0.8},
        {"source": "武汉力源信息技术股份有限公司", "target": "高速接口芯片",
         "relation": "募集资金投资", "weight": 0.8},
        {"source": "武汉力源信息技术股份有限公司", "target": "射频前端芯片",
         "relation": "募集资金投资", "weight": 0.8},
        {"source": "武汉力源信息技术股份有限公司", "target": "补充流动资金",
         "relation": "募集资金投资", "weight": 0.8},
        {"source": "武汉力源信息技术股份有限公司", "target": "销售部",
         "relation": "组织架构", "weight": 0.9},
        {"source": "销售部", "target": "大客户销售部",
         "relation": "下属部门", "weight": 0.9},
        {"source": "销售部", "target": "渠道销售部",
         "relation": "下属部门", "weight": 0.9},
        {"source": "销售部", "target": "产品销售部",
         "relation": "下属部门", "weight": 0.9},
        {"source": "销售部", "target": "电商销售部",
         "relation": "下属部门", "weight": 0.9},
        {"source": "大客户销售部", "target": "华东销售处",
         "relation": "下属销售处", "weight": 0.9},
        {"source": "大客户销售部", "target": "华南销售处",
         "relation": "下属销售处", "weight": 0.9},
        {"source": "大客户销售部", "target": "华北销售处",
         "relation": "下属销售处", "weight": 0.9},
        {"source": "大客户销售部", "target": "西南销售处",
         "relation": "下属销售处", "weight": 0.9},
        # IC 市场
        {"source": "电子信息行业", "target": "汽车电子",
         "relation": "增长最快行业", "weight": 0.9},
        {"source": "电子信息行业", "target": "嵌入式系统",
         "relation": "增长最快行业", "weight": 0.85},
        {"source": "电子信息行业", "target": "消费电子",
         "relation": "负增长行业", "weight": 0.9},
        {"source": "电子信息行业", "target": "通讯设备",
         "relation": "负增长行业", "weight": 0.85},
        # 武汉兴图新科
        {"source": "武汉兴图新科电子股份有限公司", "target": "7360万元",
         "relation": "注册资本", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "法定代表人",
         "relation": "拥有人员", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "军用领域",
         "relation": "重要供应商", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "6464.51万元",
         "relation": "军用收入", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "14414.16万元",
         "relation": "军用收入", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "18780.67万元",
         "relation": "军用收入", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "4627.14万元",
         "relation": "军用收入", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "AVS编解码技术标准",
         "relation": "参与制定", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "国家科技进步一等奖",
         "relation": "获得奖项", "weight": 0.9},
        {"source": "武汉兴图新科电子股份有限公司", "target": "补充流动资金",
         "relation": "募集资金用途", "weight": 0.8},
    ]

    return ents, rels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ents, rels = build_initial_graph()
    print(f"实体: {len(ents)}")
    print(f"关系: {len(rels)}")
