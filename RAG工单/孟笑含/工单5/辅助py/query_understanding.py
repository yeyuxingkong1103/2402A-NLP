# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：Query理解模块
功能：意图识别、消歧、分解与抽象
"""

import re
from typing import List, Dict, Any


class QueryUnderstanding:
    """Query理解模块"""
    
    # 意图类型定义
    INTENT_TYPES = {
        "数值查询": ["多少", "几", "金额", "比重", "比例", "占比", "收入", "注册资本"],
        "实体查询": ["谁", "哪家", "哪个", "什么", "法定代表人", "名称"],
        "标准查询": ["标准", "技术标准", "规范", "参与制定"],
        "供应商查询": ["供应商", "客户", "上游", "下游", "企业"],
        "荣誉查询": ["获奖", "荣誉", "一等奖", "科技进步"],
        "募集资金查询": ["募集资金", "补充流动资金", "用途", "投向"]
    }
    
    def __init__(self):
        pass
    
    def recognize_intent(self, query: str) -> str:
        """识别用户问题的核心意图"""
        for intent, keywords in self.INTENT_TYPES.items():
            for keyword in keywords:
                if keyword in query:
                    return intent
        return "通用查询"
    
    def disambiguate(self, query: str) -> str:
        """处理多义词或模糊表述"""
        # 常见消歧规则
        disambiguation_rules = {
            "报告期": "报告期（2016年度、2017年度、2018年度、2019年1-6月）",
            "公司": "武汉兴图新科电子股份有限公司",
            "本次发行": "首次公开发行股票并在科创板上市"
        }
        
        for ambiguous, clarified in disambiguation_rules.items():
            if ambiguous in query and clarified not in query:
                # 仅在需要时替换，保持原意
                pass
        
        return query
    
    def decompose(self, query: str) -> List[str]:
        """将复杂问题分解为多个子问题"""
        sub_questions = []
        
        # 检测并列结构
        if "分别" in query or "和" in query or "与" in query:
            # 尝试按标点或连接词拆分
            parts = re.split(r'[，,、和与及]', query)
            if len(parts) > 1:
                sub_questions = [p.strip() + "？" for p in parts if p.strip()]
        
        # 检测时间维度
        time_patterns = re.findall(r'(20\d{2}年(?:\d{1,2}月)?|报告期内)', query)
        if len(time_patterns) > 1:
            base_query = query
            for t in time_patterns:
                sub_q = base_query.replace("分别", "").replace(t, "").strip()
                if sub_q:
                    sub_questions.append(f"{t}{sub_q}？")
        
        return sub_questions if sub_questions else [query]
    
    def extract_keywords(self, query: str) -> List[str]:
        """提取关键信息"""
        # 移除停用词
        stop_words = {"的", "了", "是", "在", "有", "和", "与", "及", "根据", "报告期内", "分别"}
        
        # 提取实体
        keywords = []
        
        # 提取公司名
        company_pattern = r'[\u4e00-\u9fa5]+(?:股份有限公司|有限公司|公司)'
        companies = re.findall(company_pattern, query)
        keywords.extend(companies)
        
        # 提取时间
        time_pattern = r'20\d{2}年(?:\d{1,2}月)?|报告期内'
        times = re.findall(time_pattern, query)
        keywords.extend(times)
        
        # 提取其他关键词
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
        for word in words:
            if word not in stop_words and word not in keywords:
                keywords.append(word)
        
        return keywords
    
    def process(self, query: str) -> Dict[str, Any]:
        """完整的Query理解流程"""
        return {
            "original_query": query,
            "intent": self.recognize_intent(query),
            "disambiguated_query": self.disambiguate(query),
            "sub_questions": self.decompose(query),
            "keywords": self.extract_keywords(query)
        }

# ==================== 工单5 新增：指代消解集成 ====================

def _add_coref_method():
    """把核心指代消解方法动态加到 QueryUnderstanding 类"""
    pass


# 动态扩展 QueryUnderstanding（保持原类不变）
import re as _re


def rewrite_with_context(self, query: str, conversation) -> str:
    """带上下文的问题改写（工单5 新增）"""
    if conversation is None:
        return query

    # 1. 指代消解
    from coreference import CoreferenceResolver
    resolver = CoreferenceResolver(conversation)
    return resolver.resolve(query)


# 把方法绑定到 QueryUnderstanding 类
QueryUnderstanding.rewrite_with_context = rewrite_with_context
