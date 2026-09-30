# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
模块：指代消解（规则法）
功能：识别问题中的指代词，用会话历史中的实体替换
"""

import re
from typing import Optional
from conversation import Conversation


class CoreferenceResolver:
    """指代消解器（规则法）"""

    # 已知公司简称 → 全称映射
    COMPANY_MAP = {
        "武汉力源": "武汉力源信息技术股份有限公司",
        "力源信息": "武汉力源信息技术股份有限公司",
        "力源": "武汉力源信息技术股份有限公司",
        "兴图新科": "武汉兴图新科电子股份有限公司",
        "武汉兴图": "武汉兴图新科电子股份有限公司",
    }

    # 指代词
    PRONOUNS = ["他", "她", "它", "这个公司", "该公司", "这家公司", "本公司", "其"]

    # 上一轮问题类型关键词（用于"那 XXX 呢？"追问）
    QUESTION_TYPE_KEYWORDS = [
        "法定代表人", "注册资本", "技术标准", "发行股数", "募集资金",
        "上市时间", "实收资本", "注册地址", "关联方",
    ]

    def __init__(self, conversation: Conversation):
        self.conv = conversation

    def resolve(self, query: str) -> str:
        """主入口：返回改写后的问题"""
        # 1. 尝试"那 XXX 呢？"追问
        result = self._resolve_followup(query)
        if result:
            return result

        # 2. 尝试指代词替换
        result = self._resolve_pronoun(query)
        if result:
            return result

        # 3. 无需改写
        return query

    def _resolve_followup(self, query: str) -> Optional[str]:
        """处理"那 XXX 呢？"追问"""
        # 模式: "那XXX呢" / "XXX呢" / "那XXX怎么样"
        patterns = [
            r"^那(.+?)呢[？?]?$",
            r"^(.+?)呢[？?]?$",
            r"^那(.+?)[怎如]",
        ]
        for pat in patterns:
            m = re.match(pat, query.strip())
            if m:
                new_company_raw = m.group(1).strip()
                new_company = self.COMPANY_MAP.get(new_company_raw, new_company_raw)
                last_qtype = self.conv.get_last_question_type()
                if last_qtype:
                    return f"{new_company}的{last_qtype}是多少？"
                return f"{new_company}的相关信息？"
        return None

    def _resolve_pronoun(self, query: str) -> Optional[str]:
        """处理"他/这个公司"等指代词"""
        last_company = self.conv.get_last_company()
        if not last_company:
            return None

        for p in self.PRONOUNS:
            if p in query:
                # 替换指代词为公司全称
                new_query = query.replace(p, last_company, 1)
                return new_query
        return None


if __name__ == "__main__":
    conv = Conversation()
    conv.set_company("武汉兴图新科电子股份有限公司")
    conv.set_question_type("法定代表人")

    resolver = CoreferenceResolver(conv)

    tests = [
        "他参与的哪个工程荣获了国家科技进步一等奖？",
        "这个公司的法定代表人是谁？",
        "那武汉力源呢？",
        "武汉兴图新科的注册资本是多少？",
    ]

    for q in tests:
        rewritten = resolver.resolve(q)
        print(f"原文：{q}")
        print(f"改写：{rewritten}")
        print()
