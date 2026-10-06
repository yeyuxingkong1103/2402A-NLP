# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
模块：会话管理（内存版）
功能：保存多轮对话历史，供指代消解使用
"""

from typing import List, Dict, Any


class Conversation:
    """单次会话"""

    def __init__(self, max_turns: int = 10):
        self.turns: List[Dict[str, Any]] = []
        self.max_turns = max_turns
        # 关键实体追踪（用于指代消解）
        self.entities: Dict[str, Any] = {
            "last_company": None,        # 上一轮的公司名
            "last_question_type": None,  # 上一轮的问题类型（"法定代表人"/"注册资本"等）
            "company_history": [],       # 出现过的所有公司
        }

    def add(self, question: str, answer: str, query_info: Dict = None):
        """添加一轮对话"""
        self.turns.append({
            "question": question,
            "answer": answer,
            "query_info": query_info or {},
        })
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context(self, n: int = 3) -> List[Dict]:
        """获取最近 n 轮上下文"""
        return self.turns[-n:]

    def get_last_company(self) -> str:
        """获取最近讨论的公司"""
        return self.entities.get("last_company")

    def set_company(self, company: str):
        """记录当前讨论的公司"""
        self.entities["last_company"] = company
        if company and company not in self.entities["company_history"]:
            self.entities["company_history"].append(company)

    def get_last_question_type(self) -> str:
        """获取上一轮问题类型"""
        return self.entities.get("last_question_type")

    def set_question_type(self, qtype: str):
        """记录问题类型"""
        self.entities["last_question_type"] = qtype

    def clear(self):
        """清空会话"""
        self.turns = []
        self.entities = {
            "last_company": None,
            "last_question_type": None,
            "company_history": [],
        }


if __name__ == "__main__":
    conv = Conversation()
    conv.add("武汉兴图新科注册资本是多少？", "5,520万", {"intent": "数值查询"})
    conv.set_company("武汉兴图新科电子股份有限公司")
    conv.set_question_type("注册资本")

    print(f"轮数：{len(conv.turns)}")
    print(f"最近公司：{conv.get_last_company()}")
    print(f"最近问题类型：{conv.get_last_question_type()}")
