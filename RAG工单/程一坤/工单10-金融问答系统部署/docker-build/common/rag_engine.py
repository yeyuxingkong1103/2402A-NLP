# -*- coding: utf-8 -*-
"""
RAG 问答引擎
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：端到端问答流水线 —— Query向量化 → 向量检索Top-K → 组装Prompt → LLM生成。
     同时提供"纯LLM直接回答"接口，用于工单01要求的 RAG vs 纯LLM 对比。
"""
import time

from config import DEFAULT_TOP_K
from ollama_client import client
from vector_store import VectorStore

# 问答 Prompt 模板：要求仅基于检索上下文作答，注明引用页码
QA_PROMPT = """你是金融文档问答助手。请仅根据下面的参考上下文回答用户问题。
要求：
1. 答案必须来自参考上下文，不要编造；上下文中没有的信息请回答"根据文档内容未找到相关信息"。
2. 回答简洁准确，可直接引用数字与专有名词。
3. 回答末尾标注信息来源页码，格式如：（来源：第X页）

【参考上下文】
{context}

【用户问题】
{question}

【回答】"""


class RAGEngine:
    """RAG 问答引擎"""

    def __init__(self, index_name):
        self.store = VectorStore.load(index_name)
        if self.store is None:
            raise RuntimeError(f"索引 {index_name} 不存在，请先运行 build_index.py 构建索引")

    def ask(self, question, top_k=DEFAULT_TOP_K, with_context=False):
        """RAG 问答
        返回: {"answer", "sources", "time_cost", "retrieved"(可选)}
        """
        t0 = time.time()
        hits = self.store.search(question, top_k=top_k)
        t_retrieval = time.time() - t0

        context = "\n\n".join(
            f"[片段{i+1} | 第{h['page']}页 | 相似度{h['score']:.3f}]\n{h['text']}"
            for i, h in enumerate(hits)
        )
        prompt = QA_PROMPT.format(context=context, question=question)

        t1 = time.time()
        answer = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            num_predict=600,
        )
        t_gen = time.time() - t1

        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {"page": h["page"], "source": h["source"], "score": round(h["score"], 4)}
                for h in hits
            ],
            "time_cost": round(time.time() - t0, 2),
            "retrieval_time": round(t_retrieval, 2),
            "generation_time": round(t_gen, 2),
        }
        if with_context:
            result["retrieved"] = hits
        return result

    # ── 纯 LLM 直接回答（无检索，用于对比） ─────────────────
    def pure_llm_ask(self, question):
        """不经过检索，直接用 LLM 回答（对比基准）"""
        prompt = (
            f"请回答以下金融文档相关问题，简洁准确：\n\n【问题】{question}\n\n【回答】"
        )
        t0 = time.time()
        answer = client.chat([{"role": "user", "content": prompt}], temperature=0.1, num_predict=400)
        return {"question": question, "answer": answer, "time_cost": round(time.time() - t0, 2)}
