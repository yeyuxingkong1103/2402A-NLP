"""工单5：上下文感知的多轮 Query 理解和检索查询改写。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class ChatModel(Protocol):
    def invoke(self, prompt: str) -> str: ...


@dataclass
class QueryPlan:
    standalone_query: str
    sub_queries: list[str] = field(default_factory=list)
    intent: str = "lookup"
    filters: dict[str, Any] = field(default_factory=dict)


class QueryUnderstanding:
    def __init__(self, llm: ChatModel | None = None):
        self.llm = llm

    def resolve_followup(self, question: str, history: list[dict[str, str]]) -> str:
        if not history or not re.search(r"它|该公司|这家公司|那它|上述|这个|其", question):
            return question
        if self.llm:
            prompt = "根据对话历史将追问改写成独立问题，只输出问题。\n" + json.dumps(history[-6:], ensure_ascii=False) + "\n追问：" + question
            return self.llm.invoke(prompt).strip()
        previous_user = next((item["content"] for item in reversed(history) if item.get("role") == "user"), "")
        return f"{previous_user}；追问：{question}" if previous_user else question

    def plan(self, question: str, history: list[dict[str, str]] | None = None) -> QueryPlan:
        standalone = self.resolve_followup(question, history or [])
        if self.llm:
            prompt = "输出JSON: {standalone_query, sub_queries, intent, filters}。不得回答问题。\n" + standalone
            data = json.loads(self.llm.invoke(prompt))
            return QueryPlan(data.get("standalone_query", standalone), data.get("sub_queries", [standalone]), data.get("intent", "lookup"), data.get("filters", {}))
        intent = "comparison" if re.search(r"比较|区别|对比", standalone) else "lookup"
        return QueryPlan(standalone, [standalone], intent)
