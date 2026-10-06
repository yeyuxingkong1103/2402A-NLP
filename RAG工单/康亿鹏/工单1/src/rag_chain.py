# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：RAG 主链路。串联 Query 理解 → 向量召回 → 重排 → 大模型生成，
          是“问答引擎”的核心实现。
"""
import time
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from src.embeddings import get_embedding_model
from src.llm import get_cached_llm
from src.query_understanding import QueryAnalysis, understand_query
from src.reranker import get_reranker
from src.utils import logger
from src.vector_store import get_vector_store

# 回答生成提示词：严格基于检索上下文，支持中英文
_RAG_PROMPT = """你是一名专业的金融文档问答助手，正在回答关于《招股说明书》的问题。

请严格遵守以下规则：
1. 只依据【参考资料】中的内容作答，不要编造或引入文档之外的信息。
2. 如果【参考资料】中没有相关信息，请回答“根据提供的文档内容，无法回答该问题”，不要猜测。
3. 回答要准确、简洁、条理清晰；涉及金额、比例、年份、名单等数据时须完整列出。
4. 使用与用户问题相同的语言作答（中文提问用中文回答，English question answers in English）。
5. 回答末尾用一行标注引用的页码，格式：参考页码：第X页、第Y页。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""

# 纯大模型对照提示词（不使用检索结果），用于验证 RAG 的增益
_PURE_LLM_PROMPT = """你是一名金融领域助手。请直接回答用户关于《招股说明书》的问题。
如果无法确定答案，请如实说明不确定。

【用户问题】
{question}

【回答】"""


@dataclass
class RAGResult:
    """一次问答的完整结果（含检索证据与耗时，便于展示与评估）。"""

    question: str
    answer: str
    contexts: list = field(default_factory=list)
    analysis: object = None
    retrieve_time: float = 0.0
    rerank_time: float = 0.0
    generate_time: float = 0.0
    total_time: float = 0.0

    @property
    def context_texts(self) -> list:
        return [c["content"] for c in self.contexts]

    @property
    def pages(self) -> list:
        pages = []
        for ctx in self.contexts:
            page = ctx.get("metadata", {}).get("page")
            if page and page not in pages:
                pages.append(page)
        return sorted(pages)


class RAGPipeline:
    """检索增强生成主链路。"""

    def __init__(self, store=None, embedding=None, reranker=None):
        self._store = store
        self._embedding = embedding
        self._reranker = reranker
        self._answer_chain = None

    # -- 懒加载，避免未使用时占用资源 --
    @property
    def store(self):
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = get_embedding_model()
        return self._embedding

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve(self, analysis: QueryAnalysis) -> list:
        """按理解后的多个子问题召回并去重，返回去重后的候选片段。"""
        merged = {}
        for query in analysis.retrieval_queries:
            vector = self.embedding.embed_query(query)
            for hit in self.store.search(vector, top_k=config.RETRIEVE_TOP_K):
                key = hit["metadata"].get("chunk_id")
                if key not in merged or hit["score"] > merged[key]["score"]:
                    merged[key] = hit
        return list(merged.values())

    def _build_context(self, hits: list) -> str:
        blocks = []
        for i, hit in enumerate(hits, 1):
            meta = hit.get("metadata", {})
            blocks.append(
                f"[片段{i}] 来源：{meta.get('source', '')} 第{meta.get('page', '?')}页 "
                f"（类型：{meta.get('type', 'text')}）\n{hit['content']}"
            )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def _get_answer_chain(self):
        if self._answer_chain is None:
            prompt = ChatPromptTemplate.from_template(_RAG_PROMPT)
            self._answer_chain = prompt | get_cached_llm(streaming=False) | StrOutputParser()
        return self._answer_chain

    def ask(self, question: str, top_n: int = None, enable_understanding: bool = True) -> RAGResult:
        """完整问答流程，返回 RAGResult。"""
        start = time.perf_counter()
        result = RAGResult(question=question)

        # 1. Query 理解
        if enable_understanding:
            analysis = understand_query(question)
        else:
            analysis = QueryAnalysis(original=question, rewritten_query=question, keywords=[question])
        result.analysis = analysis
        query_for_rerank = analysis.rewritten_query or question

        # 2. 向量召回
        t = time.perf_counter()
        candidates = self.retrieve(analysis)
        result.retrieve_time = round(time.perf_counter() - t, 3)

        # 3. 重排
        t = time.perf_counter()
        hits = self.reranker.rerank(query_for_rerank, candidates, top_n=top_n or config.RERANK_TOP_N)
        result.rerank_time = round(time.perf_counter() - t, 3)
        result.contexts = hits

        # 4. 生成
        t = time.perf_counter()
        if not hits:
            result.answer = "根据提供的文档内容，无法回答该问题（未检索到相关片段）。"
        else:
            result.answer = self._get_answer_chain().invoke(
                {"context": self._build_context(hits), "question": question}
            )
        result.generate_time = round(time.perf_counter() - t, 3)
        result.total_time = round(time.perf_counter() - start, 3)
        logger.info("问答完成：耗时 %.3fs（召回 %.3fs / 重排 %.3fs / 生成 %.3fs），命中 %d 个片段",
                    result.total_time, result.retrieve_time, result.rerank_time,
                    result.generate_time, len(hits))
        return result

    def stream_ask(self, question: str, top_n: int = None, enable_understanding: bool = True):
        """流式问答：先返回 RAGResult（不含答案），再逐段 yield 答案文本。"""
        result = RAGResult(question=question)
        start = time.perf_counter()

        analysis = understand_query(question) if enable_understanding else QueryAnalysis(
            original=question, rewritten_query=question, keywords=[question]
        )
        result.analysis = analysis

        t = time.perf_counter()
        candidates = self.retrieve(analysis)
        result.retrieve_time = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        hits = self.reranker.rerank(analysis.rewritten_query or question, candidates,
                                    top_n=top_n or config.RERANK_TOP_N)
        result.rerank_time = round(time.perf_counter() - t, 3)
        result.contexts = hits

        if not hits:
            result.answer = "根据提供的文档内容，无法回答该问题（未检索到相关片段）。"
            yield result, result.answer
            return

        prompt = ChatPromptTemplate.from_template(_RAG_PROMPT)
        chain = prompt | get_cached_llm(streaming=True) | StrOutputParser()
        buffer = []
        for piece in chain.stream({"context": self._build_context(hits), "question": question}):
            buffer.append(piece)
            yield result, piece
        result.answer = "".join(buffer)
        result.generate_time = round(time.perf_counter() - t, 3)
        result.total_time = round(time.perf_counter() - start, 3)

    # ------------------------------------------------------------------
    # 对照：仅使用大模型（不检索）
    # ------------------------------------------------------------------
    def ask_pure_llm(self, question: str) -> str:
        """不使用检索结果，直接让大模型回答（用于“RAG vs 纯 LLM”对比分析）。"""
        prompt = ChatPromptTemplate.from_template(_PURE_LLM_PROMPT)
        chain = prompt | get_cached_llm(streaming=False) | StrOutputParser()
        try:
            return chain.invoke({"question": question})
        except Exception as exc:
            logger.error("纯 LLM 回答失败：%s", exc)
            return f"[纯 LLM 回答失败] {exc}"
