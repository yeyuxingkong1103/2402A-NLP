# -*- coding: utf-8 -*-
"""
工单十五：RAG 问答引擎（跨模态检索优化版）
  - 纯文本问题：BM25+向量混合检索 → 重排 → DeepSeek LLM
  - 图文问题：检测视觉引用 → 注入图纸结构化描述 → 检索 → LLM（结合图纸描述推理）
"""
import time
from typing import Any, Dict, List
from loguru import logger

from src.llm_client import chat
from src.query_router import route_query
from src.retriever import Retriever

RAG_SYSTEM = """你是一名专业的专利文献分析助手，擅长研读专利说明书并回答问题。
请严格遵循：
1. 仅基于提供的【参考资料】和【图纸描述】回答，不要编造资料中没有的信息。
2. 如果资料不足以回答，请说明"参考资料中未提及"。
3. 回答要准确、简洁，直接给出结论，不要展示推理过程。"""

RAG_PROMPT = """【参考资料】
{context}

【问题】
{query}

请基于以上参考资料回答问题。"""

# 跨模态 Prompt：当检索到图像信息时，明确指示模型结合图纸描述分析
CROSSMODAL_SYSTEM = """你是一名专业的专利分析助手，擅长结合专利说明书文本与技术图纸回答问题。
核心规则：
1. 专利说明书文本与图纸描述是权威依据，所有结论必须从中推理得出。
2. 当问题涉及图纸中的部件位置、物料流向、调节操作时，以【图纸描述】中的部件关系和流向为准。
3. 回答要准确、简洁，直接给出结论，不要展示推理过程。"""

CROSSMODAL_PROMPT = """【专利说明书文本】
{context}

【技术图纸描述（图{fig_no}，第{page}页）】
{fig_desc}

【问题】
{query}

请结合说明书文本与图纸描述，直接给出结论。"""


class RAGEngine:
    def __init__(self, top_k=5, max_context_chars=6000):
        self.retriever = Retriever(top_k=top_k)
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.page_images = {}
        self.page_texts = {}

    def set_page_images(self, page_images: Dict[int, str]):
        self.page_images = page_images

    def set_page_texts(self, page_texts: Dict[int, str]):
        self.page_texts = page_texts

    def ask(self, query: str) -> Dict[str, Any]:
        t0 = time.time()
        route = route_query(query)
        if route["needs_vlm"] or route["has_visual_ref"]:
            result = self._ask_crossmodal(query, route)
        else:
            result = self._ask_text(query)
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        result["route"] = route
        return result

    def _ask_text(self, query: str) -> Dict[str, Any]:
        chunks = self.retriever.retrieve(query, top_k=self.top_k)
        if not chunks:
            return {"mode": "text_rag", "answer": "抱歉，未能在知识库中找到相关信息。", "references": []}
        context = self._format_context(chunks)
        prompt = RAG_PROMPT.format(context=context, query=query)
        llm = chat(messages=[{"role": "system", "content": RAG_SYSTEM},
                             {"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=512)
        references = [{"page": c.get("page"), "chunk_id": c.get("chunk_id"),
                       "score": c.get("score"), "preview": c.get("content", "")[:100]} for c in chunks]
        return {"mode": "text_rag", "answer": llm["content"], "references": references,
                "token_usage": llm.get("token_usage", {})}

    def _ask_crossmodal(self, query: str, route: Dict) -> Dict[str, Any]:
        """图文问题：检索文本 + 注入图纸描述 → LLM 综合推理"""
        chunks = self.retriever.retrieve(query, top_k=self.top_k)
        context = self._format_context(chunks) if chunks else ""

        # 取图纸结构化描述
        fig_ctx = self.retriever.get_figure_context(query)
        fig_desc = fig_ctx.get("description", "")
        fig_no = fig_ctx.get("fig_no") or route.get("fig_no") or "?"
        page = fig_ctx.get("page") or route.get("page") or "?"

        # 若未命中图纸描述，回退用整页 OCR 文本
        if not fig_desc and page and page in self.page_texts:
            fig_desc = f"【第{page}页 OCR 文本】\n{self.page_texts[page][:1500]}"

        if not fig_desc and not context:
            return {"mode": "crossmodal_rag", "answer": "抱歉，无法分析该图纸。", "references": []}

        prompt = CROSSMODAL_PROMPT.format(context=context, fig_desc=fig_desc,
                                          fig_no=fig_no, page=page, query=query)
        llm = chat(messages=[{"role": "system", "content": CROSSMODAL_SYSTEM},
                             {"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=1024)
        references = [{"page": c.get("page"), "chunk_id": c.get("chunk_id"),
                       "score": c.get("score")} for c in chunks]
        return {"mode": "crossmodal_rag", "answer": llm["content"],
                "references": references, "figure": fig_ctx,
                "token_usage": llm.get("token_usage", {})}

    def _format_context(self, chunks: List[Dict]) -> str:
        parts, total = [], 0
        for i, c in enumerate(chunks, 1):
            page = c.get("page", "?")
            content = c.get("content", "").strip()
            block = f"[资料{i}] (第{page}页)\n{content}"
            if total + len(block) > self.max_context_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)
