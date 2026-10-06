# -*- coding: utf-8 -*-
"""
Query 理解模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能:
    1. 意图识别: 识别用户问题的核心意图
    2. 消歧: 处理多义词或模糊表述
    3. 分解与抽象: 将复杂问题分解为子问题, 提取关键词
"""
import re
from typing import Dict, List

# 意图关键词字典 (可扩展)
INTENT_KEYWORDS = {
    "财务数据": ["收入", "利润", "营业", "资产", "负债", "现金流", "比重",
                "占比", "金额", "万元", "亿元", "募集", "资金", "注册资本"],
    "公司基本信息": ["注册资本", "法定代表人", "公司", "成立", "上市",
                   "发行", "股本", "地址", "全称", "简称"],
    "技术/产品": ["技术", "标准", "专利", "研发", "产品", "编解码",
                "系统", "工程", "奖项", "进步奖"],
    "行业信息": ["行业", "上游", "下游", "领域", "供应链", "客户",
                "供应商", "竞争"],
    "风险": ["风险", "依赖", "集中度"],
}


def recognize_intent(query: str) -> List[str]:
    """意图识别: 基于关键词匹配返回可能的意图列表"""
    matched = []
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in query for k in keywords):
            matched.append(intent)
    return matched or ["其他"]


def extract_keywords(query: str) -> List[str]:
    """提取查询关键词 (用于增强检索)"""
    # 去除常见停用词
    stopwords = {"的", "是", "了", "在", "和", "与", "及", "等",
                 "请问", "什么", "哪些", "多少", "谁", "如何", "吗"}
    # 简单分词: 按标点与空格切分后过滤
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", query)
    keywords = [t for t in tokens if t not in stopwords and len(t) > 1]
    return keywords


def disambiguate(query: str) -> str:
    """
    消歧: 针对常见多义词进行替换/补全
    本项目针对"武汉兴图新科"等简称补全为全称, 提升检索准确率
    """
    aliases = {
        "兴图新科": "武汉兴图新科电子股份有限公司",
        "武汉兴图新科": "武汉兴图新科电子股份有限公司",
        "发行人": "武汉兴图新科电子股份有限公司",
        "公司": "武汉兴图新科电子股份有限公司",
    }
    enriched = query
    for short, full in aliases.items():
        if short in enriched and full not in enriched:
            enriched = enriched + " " + full
    return enriched


def decompose_query(query: str) -> List[str]:
    """
    分解与抽象: 将复杂问题分解为子问题
    简化实现: 按标点切分; 若无标点则直接返回原问题
    """
    parts = re.split(r"[,，;；。?？!！]", query)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [query]


def understand_query(query: str) -> Dict:
    """Query 理解的统一入口"""
    intents = recognize_intent(query)
    keywords = extract_keywords(query)
    enriched = disambiguate(query)
    sub_queries = decompose_query(query)
    return {
        "original_query": query,
        "intents": intents,
        "keywords": keywords,
        "enriched_query": enriched,
        "sub_queries": sub_queries,
    }


if __name__ == "__main__":
    q = "武汉兴图新科电子科技有限公司的注册资本是多少？"
    result = understand_query(q)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
