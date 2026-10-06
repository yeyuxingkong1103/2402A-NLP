# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/rag_engine.py — RAG 引擎核心
"""
import time
from typing import Any, Dict, List
from loguru import logger
from src.llm_client import chat
from src.query_understanding import analyze
from src.retriever import Retriever

RAG_SYSTEM = """你是一名专业的投资分析师，擅长研读招股说明书并回答问题。
请严格遵循：
1. 仅基于提供的【参考资料】回答，不要编造资料中没有的信息。
2. 如果参考资料不足以回答，请说明"参考资料中未提及"。
3. 引用资料时标注资料编号，例如 [资料1]。
4. 回答要条理清晰、准确、完整。"""

RAG_PROMPT_TEMPLATE = """【参考资料】
{context}

【问题】
{query}

请基于以上参考资料回答问题，并在答案中引用对应的资料编号。"""

PURE_LLM_SYSTEM = "你是一名专业的投资分析师。"

class RAGEngine:
    def __init__(self, retriever=None, top_k=5, max_context_chars=8000):
        self.retriever = retriever or Retriever(top_k=top_k)
        self.top_k = top_k; self.max_context_chars = max_context_chars
    def ask_rag(self, query):
        t0 = time.time()
        qu = analyze(query)
        rewritten = qu["rewritten_query"]
        t1 = time.time()
        chunks = self.retriever.retrieve(rewritten, top_k=self.top_k)
        retrieve_ms = (time.time() - t1) * 1000
        if not chunks:
            return {"mode": "rag", "answer": "抱歉，未能在知识库中找到相关信息。", "references": [],
                    "latency_ms": round((time.time()-t0)*1000, 1), "retrieved_chunks": [], "token_usage": {},
                    "query_understanding": qu}
        context = self._truncate_context(chunks)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)
        t2 = time.time()
        llm_result = chat(messages=[{"role": "system", "content": RAG_SYSTEM},
                                     {"role": "user", "content": prompt}])
        llm_ms = (time.time() - t2) * 1000
        references = [{"page": c.get("page"), "chunk_id": c.get("chunk_id"),
                       "score": c.get("score", c.get("distance")), "preview": c.get("content", "")[:100]} for c in chunks]
        total_ms = (time.time() - t0) * 1000
        return {"mode": "rag", "answer": llm_result["content"], "references": references,
                "latency_ms": round(total_ms, 1), "retrieved_chunks": chunks,
                "token_usage": llm_result["token_usage"], "query_understanding": qu,
                "breakdown": {"retrieve_ms": round(retrieve_ms, 1), "llm_ms": round(llm_ms, 1)}}
    def ask_llm(self, query):
        t0 = time.time()
        llm_result = chat(messages=[{"role": "system", "content": PURE_LLM_SYSTEM},
                                     {"role": "user", "content": query}])
        return {"mode": "pure_llm", "answer": llm_result["content"], "references": [],
                "latency_ms": round((time.time()-t0)*1000, 1), "retrieved_chunks": [],
                "token_usage": llm_result["token_usage"]}
    def _truncate_context(self, chunks):
        parts, total = [], 0
        for i, c in enumerate(chunks, 1):
            page = c.get("page", "?")
            content = c.get("content", "").strip()
            block = f"[资料{i}] (第{page}页)\n{content}"
            if total + len(block) > self.max_context_chars: break
            parts.append(block); total += len(block)
        return "\n\n".join(parts)

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv; load_dotenv()
    p = argparse.ArgumentParser(description="RAG Engine CLI（工单：人工智能NLP-RAG-基于PDF文档的问答系统）")
    p.add_argument("--query", "-q", required=True)
    p.add_argument("--mode", choices=["rag", "llm", "both"], default="both")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    engine = RAGEngine(top_k=args.top_k)
    if args.mode in ("rag", "both"):
        print("\n" + "="*60 + "\n🔎 RAG 模式\n" + "="*60)
        r = engine.ask_rag(args.query)
        print(f"⏱️  总: {r['latency_ms']:.0f}ms  检索: {r['breakdown']['retrieve_ms']:.0f}ms  LLM: {r['breakdown']['llm_ms']:.0f}ms")
        print(f"\n💡 答案:\n{r['answer']}")
        print(f"\n📚 引用 ({len(r['references'])} 条):")
        for i, ref in enumerate(r["references"], 1):
            print(f"  [{i}] 第{ref['page']}页  score={ref['score']:.4f}  {ref['preview'][:60]}...")
    if args.mode in ("llm", "both"):
        print("\n" + "="*60 + "\n🤖 纯 LLM 模式\n" + "="*60)
        r2 = engine.ask_llm(args.query)
        print(f"⏱️  {r2['latency_ms']:.0f}ms\n\n💡 答案:\n{r2['answer']}")
