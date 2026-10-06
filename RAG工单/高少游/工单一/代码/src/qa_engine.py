# -*- coding: utf-8 -*-
"""问答引擎：RAG 检索增强生成 与 纯 LLM 对比基线
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

说明：
- answer_rag：Query 理解 -> 混合检索 -> 上下文 -> 生成，返回答案及检索依据；
- answer_llm_only：不注入文档上下文，仅靠大模型自身常识/知识回答（用于对比分析）；
- 两条链路都记录耗时时长，便于验收“3 秒内响应”指标。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src import config
from src.query_understanding import Analysis, QueryUnderstanding
from src.retriever import HybridRetriever, RetrievedDoc, format_contexts

logger = logging.getLogger(__name__)

# RAG 生成提示词：要求模型严格基于引用文档作答，防止幻觉
_RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是一个严谨的金融招股说明书问答助手。
请仅依据下面提供的【参考文档】内容回答问题。
- 如果参考文档中包含答案，请直接、准确、简洁地回答，必要时给出数值或表格；
- 如果参考文档不包含答案，请明确说明“根据招股说明书内容无法回答”，不要编造；
- 回答中使用到的关键数据请注明来源页码。

【参考文档】
{context}

用户问题：{question}
请回答："""
)

# 纯 LLM 对比提示词：无检索上下文，仅靠模型知识
_LLM_ONLY_PROMPT = ChatPromptTemplate.from_template(
    """你是一个金融问答助手。请结合你自己的知识，尽可能准确、简洁地回答以下问题。
若不确定，请如实说明。

用户问题：{question}
请回答："""
)


@dataclass
class AnswerResult:
    """一次问答的结果。"""

    question: str
    answer: str
    mode: str                      # "rag" / "llm_only"
    elapsed: float = 0.0           # 耗时（秒）
    sources: List[Dict] = field(default_factory=list)  # 检索到的依据
    analysis: Optional[Analysis] = None  # Query 理解结果


class QAEngine:
    """问答引擎：提供 RAG 生成与纯 LLM 基线两种能力。"""

    def __init__(self, llm: Optional[ChatOllama] = None, retriever=None):
        self.llm = llm or ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            num_predict=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT,
        )
        self.retriever = retriever or HybridRetriever()
        self.understanding = QueryUnderstanding(llm=self.llm)

    # ---- 检索（支持子问题/实体多路召回并去重）--------------------------------
    def _retrieve_unique(self, queries: List[str], k: int) -> List[RetrievedDoc]:
        merged: Dict[str, RetrievedDoc] = {}
        for q in queries:
            for r in self.retriever.retrieve(q, k=k):
                merged.setdefault(r.doc.page_content, r)
        ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        return ranked[:k]

    # ---- RAG 问答 -------------------------------------------------------------
    def answer_rag(
        self, question: str, use_understanding: bool = True
    ) -> AnswerResult:
        """RAG 问答：Query理解 -> 检索 -> 生成。"""
        start = time.time()
        analysis = self.understanding.analyze(question) if use_understanding else None

        retrieval_queries = (
            analysis.retrieval_queries if analysis else [question]
        )
        results = self._retrieve_unique(retrieval_queries, k=config.TOP_K)
        context = format_contexts(results)
        sources = [
            {
                "page": r.doc.metadata.get("page", "?"),
                "score": round(r.score, 4),
                "text": r.doc.page_content[:400],
            }
            for r in results
        ]

        prompt = _RAG_PROMPT.format(context=context, question=question)
        answer = self._invoke(prompt)
        return AnswerResult(
            question=question,
            answer=answer,
            mode="rag",
            elapsed=time.time() - start,
            sources=sources,
            analysis=analysis,
        )

    # ---- 纯 LLM 对比基线 --------------------------------------------------------
    def answer_llm_only(self, question: str) -> AnswerResult:
        """纯 LLM 问答：不注入检索上下文。"""
        start = time.time()
        prompt = _LLM_ONLY_PROMPT.format(question=question)
        answer = self._invoke(prompt)
        return AnswerResult(
            question=question, answer=answer, mode="llm_only",
            elapsed=time.time() - start,
        )

    def _invoke(self, prompt: str) -> str:
        try:
            msg = self.llm.invoke(prompt)
            text = str(getattr(msg, "content", msg))
            return text.strip()
        except Exception as exc:
            logger.error("llm invoke failed: %s", exc)
            return f"（回答失败：{exc}）"