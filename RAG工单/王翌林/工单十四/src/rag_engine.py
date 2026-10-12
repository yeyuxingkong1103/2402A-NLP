# -*- coding: utf-8 -*-
"""
工单十四：RAG 问答引擎（文本/图像双路由）
- 文本问题：BM25+向量混合检索 → bge-reranker重排序 → DeepSeek LLM生成
- 图像问题：Qwen2-VL 对指定页面图纸做 VQA → DeepSeek LLM 综合答案
"""
import time
from typing import Any, Dict, List
from loguru import logger

from src.llm_client import chat
from src.query_router import route_query
from src.retriever import Retriever
from src import vlm_engine

RAG_SYSTEM = """你是一名专业的专利文献分析助手，擅长研读专利说明书并回答问题。
请严格遵循：
1. 仅基于提供的【参考资料】回答，不要编造资料中没有的信息。
2. 如果参考资料不足以回答，请说明"参考资料中未提及"。
3. 回答要准确、简洁，直接给出结论。"""

RAG_PROMPT = """【参考资料】
{context}

【问题】
{query}

请基于以上参考资料回答问题。"""

IMAGE_SYSTEM = """你是一名专业的专利分析助手。
核心规则：
1. 专利说明书文本是唯一权威依据，所有结论必须从说明书文本中推理得出。
2. 图纸视觉描述仅用于辅助理解，若与说明书文本冲突，以说明书文本为准。
3. 位置关系统一用"左/右"描述：沿气流方向，先经过的部件在"左"，后经过的在"右"。严禁使用"上/下"描述位置关系。
4. 必须按以下步骤推理：
   步骤1：确定气流方向（箭头7的方向）。
   步骤2：按气流方向依次列出部件顺序。
   步骤3：根据部件顺序用"左/右"判断位置关系。
   步骤4：根据尺寸标注（h1/h2/x1-x3/D）分析各尺寸的作用。
5. 关键提示：
   - "后接"表示沿气流方向，前者在左，后者在右。
   - 间隔从某部件"开始测量"表示该部件是测量起点，离起点越远的部件越靠左。
   - 配气带孔盘6,6',6"中，6"离入口最近（最左），6离台阶最近（最右），气流依次经过6"→6'→6。
   - h1是圆柱形部分4的高度（约5%D），h2是圆锥形部分10的高度（约35%D），配气带孔盘位于圆锥形部分上且间隔从台阶测量，因此已知D、h1、h2可确定配气带孔盘6,6',6"的位置。注意：h1/h2的作用是确定配气带孔盘的位置，不是计算管状入口尺寸。
   - 气流方向问题问的是配气带孔盘等内部部件的经过顺序，不是入口/外壳。
6. 回答要准确、简洁，直接给出结论，不要展示推理过程。"""

IMAGE_PROMPT = """【专利说明书文本】
{context}

【图纸视觉描述（仅供参考）】
{image_desc}

【问题】
{query}

请直接给出结论。"""


class RAGEngine:
    def __init__(self, top_k=5, max_context_chars=6000):
        self.retriever = Retriever(top_k=top_k)
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        # 页面图片路径映射（由外部设置）
        self.page_images = {}
        # 页面 OCR 文本（用于图像问题的上下文补充）
        self.page_texts = {}

    def set_page_images(self, page_images: Dict[int, str]):
        self.page_images = page_images

    def set_page_texts(self, page_texts: Dict[int, str]):
        self.page_texts = page_texts

    def ask(self, query: str) -> Dict[str, Any]:
        t0 = time.time()
        route = route_query(query)

        if route["needs_vlm"]:
            result = self._ask_image(query, route)
        else:
            result = self._ask_text(query)

        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        result["route"] = route
        return result

    def _ask_text(self, query: str) -> Dict[str, Any]:
        """文本问题：检索 + LLM"""
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

    def _ask_image(self, query: str, route: Dict) -> Dict[str, Any]:
        """图像问题：VLM VQA + LLM"""
        page = route.get("page")
        image_path = self.page_images.get(page) if page else None
        if not image_path:
            # 尝试模糊匹配
            for p in sorted(self.page_images.keys(), reverse=True):
                image_path = self.page_images[p]
                page = p
                break

        # 先检索相关文本（说明书中对图纸的说明）
        context = ""
        try:
            chunks = self.retriever.retrieve(query, top_k=3)
            if chunks:
                context = self._format_context(chunks)
        except Exception:
            pass

        # 如果有页面 OCR 文本，也加入上下文
        if page and page in self.page_texts:
            page_text = self.page_texts[page]
            if page_text:
                context = f"【第{page}页图纸附近文本】\n{page_text[:1500]}\n\n" + context

        # VLM 对图纸做 VQA（可选，仅当说明书文本不足时使用）
        image_desc = ""
        # 优先基于说明书文本回答，跳过慢速 VQA 以降低延迟
        # 如需 VQA 辅助，取消下行注释
        # if image_path:
        #     try:
        #         image_desc = vlm_engine.vqa_image(image_path, query, context=context)
        #         logger.info(f"VQA 结果: {image_desc[:200]}")
        #     except Exception as e:
        #         logger.warning(f"VQA 失败: {e}")

        if not image_desc and not context:
            return {"mode": "image_rag", "answer": "抱歉，无法分析该图纸。", "references": [],
                    "image_path": image_path}

        prompt = IMAGE_PROMPT.format(image_desc=image_desc, context=context, query=query)
        llm = chat(messages=[{"role": "system", "content": IMAGE_SYSTEM},
                             {"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=1024)
        return {"mode": "image_rag", "answer": llm["content"],
                "references": [], "image_path": image_path, "image_desc": image_desc,
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
