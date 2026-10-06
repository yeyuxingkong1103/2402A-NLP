# -*- coding: utf-8 -*-
"""Query 理解模块：意图识别、消歧、分解与抽象
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

说明：
- 意图识别：判断问题属于“事实查询 / 财务数据 / 计算 / 归属关系 / 其他”等类型；
- 消歧：对多义词、行业术语、模糊表述进行规范化重写，提升检索命中率；
- 分解与抽象：把复杂 / 复合问题拆分为若干子问题，并抽取关键实体关键词用于检索；
- 服务对象：qa_engine 在检索前调用本模块对 Query 进行增强。

实现：基于本地 ChatOllama + structured 提示词，解析为 JSON。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_ollama import ChatOllama

from src import config

logger = logging.getLogger(__name__)

_INTENT_HINT = (
    "intent: 意图类型（fact查询事实/finance财务数据/calc计算/relation归属关系/other其他）"
)

_ANALYZE_PROMPT = """你是金融招股说明书问答系统中的 Query 理解模块。
请对用户问题执行：1) 意图识别 2) 多义词消歧、规范术语 3) 复杂问题分解为子问题 4) 抽取检索关键词。

严格只输出一个 JSON 对象，不要输出任何其他文字，格式如下：
{
  "intent": "{intent_type}",
  "disambiguated_query": "消歧/规范化后的完整问题",
  "entities": ["关键实体词1", "关键实体词2"],
  "sub_questions": ["子问题1", "子问题2"]
}
（若非复杂问题，sub_questions 可只含 1 个元素，即 disambiguated_query。）

用户问题：{question}
"""


@dataclass
class Analysis:
    """Query 理解结果。"""

    question: str
    intent: str = "other"
    disambiguated_query: str = ""
    entities: List[str] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)

    @property
    def retrieval_queries(self) -> List[str]:
        """用于检索的候选查询：优先消歧后的问题，其次子问题，最后原问题。"""
        subs = [s for s in self.sub_questions if s]
        if subs:
            return subs
        dq = self.disambiguated_query if self.disambiguated_query else self.question
        return [dq]


def _parse_json(raw: str) -> Optional[dict]:
    """从模型输出中稳健地提取 JSON 对象。"""
    if not raw:
        return None
    text = raw.strip()
    # 去掉 ```json ... ``` 包裹
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # 提取首个 { ... } 对象
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class QueryUnderstanding:
    """Query 理解服务。"""

    def __init__(self, llm: Optional[ChatOllama] = None):
        self.llm = llm or ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            num_predict=256,           # 分析任务生成尽可能短
        )

    def analyze(self, question: str) -> Analysis:
        """对用户问题做意图识别、消歧、分解与抽象。"""
        if not question or not question.strip():
            return Analysis(question=question or "")

        prompt = _ANALYZE_PROMPT.replace("{question}", question or "")
        try:
            raw = self.llm.invoke(prompt)
            data = _parse_json(str(raw.content))
        except Exception as exc:
            logger.warning("query understanding failed, fallback to raw: %s", exc)
            data = None

        if not data:
            return Analysis(question=question, disambiguated_query=question)

        entities = data.get("entities") or []
        subs = data.get("sub_questions") or []
        return Analysis(
            question=question,
            intent=data.get("intent", "other"),
            disambiguated_query=data.get("disambiguated_query") or question,
            entities=self._clean(entities),
            sub_questions=self._clean(subs),
        )

    @staticmethod
    def _clean(items: List[str]) -> List[str]:
        return [str(x).strip() for x in items if str(x).strip()]