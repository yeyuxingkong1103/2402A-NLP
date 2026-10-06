# -*- coding: utf-8 -*-
"""
查询增强模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

功能:
    1. 查询扩展: 同义词补充, 扩大召回
    2. 关键词加权: 识别核心实体
    3. 多语言支持: 英文问题映射
    4. 检索策略选择: 根据意图调整参数
"""
import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# 金融/招股说明书领域同义词字典
SYNONYM_MAP = {
    "收入": ["收入", "营收", "营业额", "销售额", "营业收入"],
    "营业收入": ["收入", "营收", "营业额", "销售额", "营业收入"],
    "占比": ["占比", "比重", "比例", "份额"],
    "比重": ["占比", "比重", "比例", "份额"],
    "注册资本": ["注册资本", "股本", "资本", "实收资本"],
    "法定代表人": ["法定代表人", "法人代表", "董事长", "负责人"],
    "技术标准": ["技术标准", "标准", "规范", "国家标准", "行业标准"],
    "军用": ["军用", "军方", "军队", "国防", "军工"],
    "行业": ["行业", "产业", "领域"],
    "上游": ["上游", "供应链", "供应商", "原材料"],
    "下游": ["下游", "客户端", "客户", "市场"],
    "工程": ["工程", "项目", "系统", "系统工程"],
    "进步奖": ["进步奖", "奖项", "科技进步", "一等奖"],
    "募集资金": ["募集资金", "融资", "发行募集", "募集"],
    "流动资金": ["流动资金", "营运资金", "现金", "运营资金"],
    "研发": ["研发", "研究开发", "技术开发", "产品开发"],
    "产品": ["产品", "设备", "系统", "方案", "解决方案"],
    "公司": ["公司", "发行人", "企业", "本公司", "股份有限公司"],
}

# 英文关键词 → 中文映射 (多语言支持)
EN_TO_ZH_MAP = {
    "revenue": "收入",
    "income": "收入",
    "sales": "收入",
    "ratio": "占比",
    "proportion": "占比",
    "registered capital": "注册资本",
    "legal representative": "法定代表人",
    "technology standard": "技术标准",
    "military": "军用",
    "industry": "行业",
    "upstream": "上游",
    "downstream": "下游",
    "project": "工程",
    "award": "进步奖",
    "fund": "募集资金",
    "working capital": "流动资金",
    "rd": "研发",
    "product": "产品",
    "company": "公司",
}


def _is_english(query: str) -> bool:
    """检测是否为英文问题 (简单启发式)"""
    eng_chars = sum(1 for c in query if 'a' <= c.lower() <= 'z')
    total_chars = len(query.replace(" ", ""))
    return total_chars > 0 and eng_chars / total_chars > 0.5


def translate_en_query(query: str) -> str:
    """将英文问题翻译/映射为中文关键词"""
    if not _is_english(query):
        return query
    translated = query
    for en, zh in EN_TO_ZH_MAP.items():
        translated = re.sub(en, zh, translated, flags=re.IGNORECASE)
    logger.info(f"[多语言] 英文问题映射: '{query}' → '{translated}'")
    return translated


def expand_query(query: str) -> Tuple[str, List[str]]:
    """
    查询扩展: 补充同义词

    Returns:
        (扩展后的查询字符串, 额外添加的同义词列表)
    """
    # 先翻译英文
    query_zh = translate_en_query(query)

    # 提取查询中的关键词
    import jieba
    try:
        tokens = list(jieba.cut(query_zh))
    except ImportError:
        tokens = list(query_zh)

    extra_terms = []
    for t in tokens:
        for key, syns in SYNONYM_MAP.items():
            if t == key or t in syns:
                for s in syns:
                    if s not in extra_terms and s not in query_zh:
                        extra_terms.append(s)
                break

    expanded = query_zh
    if extra_terms:
        expanded = query_zh + " " + " ".join(extra_terms)
        logger.info(f"[查询扩展] 原查询 '{query_zh}' → 扩展 '{expanded}' "
                     f"(+{len(extra_terms)} 同义词)")

    return expanded, extra_terms


def detect_intent(query: str) -> str:
    """意图识别 (决定检索参数)"""
    financial = ["收入", "利润", "占比", "比重", "募集", "资金", "注册资本", "万元", "亿元"]
    company = ["法定代表人", "成立", "上市", "地址", "全称"]
    tech = ["技术", "标准", "专利", "研发", "编解码", "工程", "奖项"]
    industry = ["行业", "上游", "下游", "领域", "供应商"]

    if any(k in query for k in financial):
        return "financial"
    if any(k in query for k in company):
        return "company"
    if any(k in query for k in tech):
        return "tech"
    if any(k in query for k in industry):
        return "industry"
    return "general"


def get_search_params(query: str) -> Dict:
    """根据意图返回检索参数"""
    intent = detect_intent(query)
    # 默认参数
    params = {
        "intent": intent,
        "top_k": 5,
        "raw_k": 20,
    }
    # 财务/行业类问题需要更大的 Top-K (可能涉及多页数据)
    if intent in ("financial", "industry"):
        params["top_k"] = 6
        params["raw_k"] = 25
    # 公司/技术类问题更精准
    if intent in ("company", "tech"):
        params["top_k"] = 4
        params["raw_k"] = 15
    return params


def enhance_query(query: str) -> Dict:
    """查询增强统一入口"""
    expanded, extras = expand_query(query)
    params = get_search_params(query)
    return {
        "original": query,
        "expanded": expanded,
        "extra_terms": extras,
        "intent": params["intent"],
        "top_k": params["top_k"],
        "raw_k": params["raw_k"],
        "is_english": _is_english(query),
    }


if __name__ == "__main__":
    tests = [
        "武汉兴图新科电子股份有限公司的注册资本是多少?",
        "报告期内来自军用领域的收入占主营业务的比重是多少?",
        "What is the registered capital of the company?",
        "武汉兴图新科参与制定了哪个技术标准?",
    ]
    for q in tests:
        r = enhance_query(q)
        print(f"\n原查询: {q}")
        print(f"  扩展: {r['expanded']}")
        print(f"  同义词: {r['extra_terms']}")
        print(f"  意图: {r['intent']} | top_k={r['top_k']} | raw_k={r['raw_k']}")
