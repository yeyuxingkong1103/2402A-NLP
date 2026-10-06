# -*- coding: utf-8 -*-
"""问答引擎（优化版）：优化链路 / 基线链路 双通道
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

提供四条链路，用于「优化前后」对比与消融分析：

| 链路 | 检索 | 答案合成 | 用途 |
|---|---|---|---|
| baseline            | 基线检索（固定切片 + RRF） | 本地 LLM 生成 | 复刻 01 工单，优化前基准 |
| baseline_extractive | 基线检索                   | 抽取式合成   | 消融：仅换答案合成 |
| optimized           | 优化检索（结构分块+多路召回+重排） | 抽取式合成 | 本工单最终方案 |
| optimized_llm       | 优化检索                   | 本地 LLM 生成 | 消融：仅换检索 |

所有链路均记录端到端耗时，用于验收“≤3s”指标。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src import config
from src.answer_builder import build_answer
from src.query_understanding import QueryAnalysis, QueryUnderstanding
from src.retriever import (BaselineRetriever, HybridRetriever, format_contexts)

logger = logging.getLogger(__name__)

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
    mode: str
    elapsed: float = 0.0
    sources: List[Dict] = field(default_factory=list)
    analysis: Optional[QueryAnalysis] = None
    citations: List[int] = field(default_factory=list)


class QAEngine:
    """问答引擎：优化链路与基线链路。"""

    def __init__(self, llm: Optional[ChatOllama] = None,
                 opt_retriever: Optional[HybridRetriever] = None,
                 base_retriever: Optional[BaselineRetriever] = None):
        self.llm = llm or ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            num_predict=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT,
        )
        self.understanding = QueryUnderstanding()
        self._opt = opt_retriever
        self._base = base_retriever

    # ---- 懒加载检索器（避免无关链路也加载向量库） ---------------------------
    @property
    def opt_retriever(self) -> HybridRetriever:
        if self._opt is None:
            self._opt = HybridRetriever()
        return self._opt

    @property
    def base_retriever(self) -> BaselineRetriever:
        if self._base is None:
            from src.knowledge_base import load_kb

            store = load_kb(config.BASE_DB_DIR)
            self._base = BaselineRetriever(store=store)
        return self._base

    # ---- 通用：把检索结果转成 sources --------------------------------------
    @staticmethod
    def _sources_from(scored, limit: int = 6) -> List[Dict]:
        out = []
        for r in scored[:limit]:
            doc = r.doc if hasattr(r, "doc") else r
            out.append({
                "page": doc.metadata.get("page", "?"),
                "score": round(float(getattr(r, "score", 0.0)), 4),
                "ctype": doc.metadata.get("ctype", "text"),
                "section": doc.metadata.get("section", ""),
                "text": doc.page_content[:400],
                "detail": getattr(r, "detail", None),
            })
        return out

    # ================= 优化链路 =============================================
    def answer_optimized(self, question: str) -> AnswerResult:
        """优化链路：Query 理解 → 多路混合召回 → 重排 → 抽取式答案合成。"""
        start = time.time()
        analysis = self.understanding.analyze(question)
        pool = self.opt_retriever.retrieve(analysis, k=config.ANSWER_POOL)
        bundle = build_answer(analysis, pool)
        return AnswerResult(
            question=question, answer=bundle.answer, mode="optimized",
            elapsed=time.time() - start, sources=self._sources_from(pool, limit=config.TOP_K),
            analysis=analysis, citations=bundle.citations,
        )

    def answer_optimized_llm(self, question: str) -> AnswerResult:
        """优化检索 + 本地 LLM 生成（消融对照）。"""
        start = time.time()
        analysis = self.understanding.analyze(question)
        scored = self.opt_retriever.retrieve(analysis, k=config.TOP_K)
        context = format_contexts(scored)
        prompt = _RAG_PROMPT.format(context=context, question=question)
        answer = self._invoke(prompt)
        return AnswerResult(
            question=question, answer=answer, mode="optimized_llm",
            elapsed=time.time() - start, sources=self._sources_from(scored),
            analysis=analysis,
        )

    # ================= 基线链路 =============================================
    def answer_baseline(self, question: str) -> AnswerResult:
        """基线链路：基线检索 + 本地 LLM 生成（复刻 01 工单）。"""
        start = time.time()
        results = self.base_retriever.retrieve(question, k=config.BASE_TOP_K)
        context = format_contexts(results)
        prompt = _RAG_PROMPT.format(context=context, question=question)
        answer = self._invoke(prompt)
        return AnswerResult(
            question=question, answer=answer, mode="baseline",
            elapsed=time.time() - start, sources=self._sources_from(results),
        )

    def answer_baseline_extractive(self, question: str) -> AnswerResult:
        """基线检索 + 抽取式合成（消融：隔离检索侧优化贡献）。"""
        start = time.time()
        analysis = self.understanding.analyze(question)
        results = self.base_retriever.retrieve(question, k=config.BASE_TOP_K)
        from src.reranker import ScoredDoc

        pseudo = [ScoredDoc(r.doc, r.score, {}) for r in results]
        bundle = build_answer(analysis, pseudo)
        return AnswerResult(
            question=question, answer=bundle.answer, mode="baseline_extractive",
            elapsed=time.time() - start, sources=self._sources_from(results),
            analysis=analysis, citations=bundle.citations,
        )

    # ---- 纯 LLM 基线（无检索） ----------------------------------------------
    def answer_llm_only(self, question: str) -> AnswerResult:
        start = time.time()
        prompt = _LLM_ONLY_PROMPT.format(question=question)
        answer = self._invoke(prompt)
        return AnswerResult(question=question, answer=answer, mode="llm_only",
                            elapsed=time.time() - start)

    # ---- LLM 调用（容错） ---------------------------------------------------
    def _invoke(self, prompt: str) -> str:
        try:
            msg = self.llm.invoke(prompt)
            return str(getattr(msg, "content", msg)).strip()
        except Exception as exc:
            logger.error("llm invoke failed: %s", exc)
            return f"（回答失败：{exc}）"


# ---- 检索侧指标（用于隔离评估“检索准确率”的提升） ---------------------------
def retrieval_hit(results, key_tokens: List[str]) -> bool:
    """检索到的上下文中是否包含参考答案关键词（Recall 命中）。"""
    text = " ".join(
        (r.doc if hasattr(r, "doc") else r).page_content for r in results
    ).replace(" ", "")
    return any(t.replace(" ", "") in text for t in key_tokens) if key_tokens else False