# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：Query 理解模块。对用户问题进行意图识别、消歧、分解与抽象，
          输出规范化的检索用问题，是“Query 理解”功能需求的实现。
"""
import json
import re
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

import config
from src.llm import get_cached_llm
from src.utils import logger

# 意图类别（面向招股说明书/金融文档问答场景）
INTENT_TYPES = ["事实查询", "数据统计", "对比分析", "原因分析", "定义解释", "总结归纳", "其他"]

_QUERY_UNDERSTANDING_PROMPT = """你是一个金融文档问答系统的“查询理解”模块，处理的文档是《招股说明书》。
请分析用户问题，并严格输出一个 JSON 对象（不要输出任何多余文字、不要使用 Markdown 代码块）。

JSON 字段要求：
- language: 问题语言，"zh" 或 "en"
- intent: 问题意图，只能从以下类别中选择一个：{intent_types}
- rewritten_query: 消歧与规范化后的完整问题（补全指代、明确主体，如把“该公司”补全为具体公司名称）；若原问题已清晰则原样输出
- sub_questions: 若问题是多跳/复合问题，拆解为 0~{max_sub} 个可独立检索的子问题；若无需拆解则为空数组
- keywords: 从问题中抽取的关键实体与检索关键词（3~8 个）

用户问题：{question}

JSON 输出："""


@dataclass
class QueryAnalysis:
    """Query 理解结果。"""

    original: str
    language: str = "zh"
    intent: str = "其他"
    rewritten_query: str = ""
    sub_questions: list = field(default_factory=list)
    keywords: list = field(default_factory=list)

    @property
    def retrieval_queries(self) -> list:
        """实际用于检索的问题列表。

        工单02 检索优化（多路召回）：同时使用「原始问题」与「消歧后的规范化问题」检索，
        两者互为补充——规范化问题补充上下文，原始问题保留用户原始措辞，
        避免改写偏离导致召回退化；再并入分解出的子问题。
        """
        queries = []
        for query in (self.original, self.rewritten_query):
            if query and query not in queries:
                queries.append(query)
        for sub in self.sub_questions:
            if sub and sub not in queries:
                queries.append(sub)
        return queries[: config.MAX_SUB_QUESTIONS + 2]


def _extract_json(text: str) -> dict:
    """从模型输出中稳健地提取 JSON（兼容代码块包裹等情形）。"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            text = match.group(0)
    return json.loads(text)


def understand_query(question: str, llm=None) -> QueryAnalysis:
    """对用户问题进行理解。任何异常都会降级为“原问题直检索”，保证系统可用。"""
    question = (question or "").strip()
    if not question:
        return QueryAnalysis(original="", rewritten_query="")

    fallback = QueryAnalysis(original=question, rewritten_query=question, keywords=[question])
    if not config.QUERY_UNDERSTANDING_ENABLED:
        return fallback

    try:
        llm = llm or get_cached_llm(streaming=False)
        prompt = ChatPromptTemplate.from_template(_QUERY_UNDERSTANDING_PROMPT).format(
            intent_types="、".join(INTENT_TYPES),
            max_sub=config.MAX_SUB_QUESTIONS,
            question=question,
        )
        content = llm.invoke(prompt).content
        data = _extract_json(content)

        analysis = QueryAnalysis(
            original=question,
            language=data.get("language", "zh"),
            intent=data.get("intent", "其他"),
            rewritten_query=data.get("rewritten_query") or question,
            sub_questions=[q for q in (data.get("sub_questions") or []) if isinstance(q, str)],
            keywords=[k for k in (data.get("keywords") or []) if isinstance(k, str)],
        )
        logger.info("Query 理解：intent=%s, 子问题=%d, 关键词=%s",
                    analysis.intent, len(analysis.sub_questions), analysis.keywords)
        return analysis
    except Exception as exc:
        logger.warning("Query 理解失败，降级为原问题检索：%s", exc)
        return fallback
