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

# ================= 工单二：优化链路（人工智能NLP-RAG-基于PDF文档的问答系统优化） =================
# HybridRetriever + 父子块召回 + 多语言路由 + 语义缓存，进程级懒加载单例
_HYBRID = None
_PARENT_BY_ID = {}


def get_hybrid_retriever():
    """工单二优化检索器懒加载单例（子块检索索引 + 父块内容映射）"""
    global _HYBRID, _PARENT_BY_ID
    if _HYBRID is None:
        import json
        from src.hybrid_retriever import HybridRetriever
        path = "data/optimized/招股说明书1_chunks_optimized.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _PARENT_BY_ID = {p["chunk_id"]: p for p in data.get("parent_chunks", [])}
        _HYBRID = HybridRetriever(chunks=data.get("chunks", []), use_rerank=True)
        logger.info(f"✅ 工单二优化检索器就绪: {len(data.get('chunks', []))} 子块 / {len(_PARENT_BY_ID)} 父块")
    return _HYBRID


def _expand_to_parents(results: List[Dict], max_chars: int = 9000) -> List[Dict]:
    """工单二父子块召回：检索命中的子块 → 父块内容（small-to-big），按 chunk_id 去重；
    表格键值专项召回（kv=True）的父块置顶（如发行人基本情况表，人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    seen, out, total = set(), [], 0
    for r in results:
        pid = r.get("parent_id")
        p = _PARENT_BY_ID.get(pid) if pid else None
        content = p["text"] if p else r.get("content", "")
        cid = pid or r["chunk_id"]
        if cid in seen:
            continue
        if total + len(content) > max_chars:
            continue
        seen.add(cid)
        total += len(content)
        out.append({"chunk_id": cid, "page": (p or {}).get("page", r.get("page")),
                    "content": content, "score": r.get("score"),
                    "heading": (p or {}).get("heading", r.get("heading")),
                    "kv": bool(r.get("kv", False))})
    out.sort(key=lambda x: (not x["kv"], -(x["score"] or 0)))  # kv 块置顶，其余按分数
    return out if out else results


class OptimizedEngine:
    """工单二优化问答引擎（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    语义缓存 lookup → 双语检索（翻译路由）→ 父子块召回 → LLM 生成 → 缓存 put；
    latency 全程分解，供 /api/ask 与 evaluate.py 使用"""

    def __init__(self, top_k: int = 8):  # 工单二：top_k=8 提升表格键值块召回（如"法定代表人：程家明"）
        self.top_k = top_k

    def ask(self, query: str, lang_override=None, use_cache: bool = True,
            top_k: int = None) -> Dict[str, Any]:
        from src.cache import get_semantic_cache
        from src.multilingual import (bilingual_retrieve, build_system_prompt,
                                      _format_context, detect_language)
        t0 = time.time()
        top_k = top_k or self.top_k
        # 1) 语义缓存（相似问题直接复用答案，延迟 ~10-20ms；lang 隔离防止跨语言误命中）
        cache = get_semantic_cache()
        query_lang = lang_override or detect_language(query)
        if use_cache:
            hit = cache.lookup(query, lang=query_lang)
            if hit:
                # lookup 返回扁平结构：payload 字段 + cache_hit/similarity
                payload = {k: v for k, v in hit.items()
                           if k in ("answer", "references", "lang")}
                return {"mode": "rag_optimized", "answer": payload.get("answer", ""),
                        "references": payload.get("references", []),
                        "latency_ms": round((time.time() - t0) * 1000, 1),
                        "cache_hit": True, "cache_similarity": hit.get("similarity"),
                        "token_usage": {}, "lang": payload.get("lang"), "translated": False}
        # 2) 双语检索（en 自动翻译路由，失败回退跨语言）
        retriever = get_hybrid_retriever()
        r = bilingual_retrieve(retriever, query, top_k=top_k, lang_override=lang_override)
        retrieve_ms = (time.time() - t0) * 1000
        if not r["results"]:
            return {"mode": "rag_optimized", "answer": "抱歉，未能在知识库中找到相关信息。",
                    "references": [], "latency_ms": round((time.time() - t0) * 1000, 1),
                    "cache_hit": False, "lang": r["lang"], "translated": r["translated"]}
        # 3) 父子块召回：子块命中 → 父块上下文；表格键值专项块单独高亮（人工智能NLP-RAG-基于PDF文档的问答系统优化）
        parents = _expand_to_parents(r["results"])
        kv_ctx = ""
        kv_results = [x for x in r["results"] if x.get("kv")]
        if kv_results:
            kv_parents = _expand_to_parents(kv_results, max_chars=2500)
            if kv_parents:
                kv_ctx = "【与问题直接相关的键值信息表——最高优先级，其中的发行人键值信息可信】\n" + \
                         "\n---\n".join(f"(page={p['page']}) {p['content'][:1200]}"
                                        for p in kv_parents) + "\n\n"
        context = kv_ctx + _format_context([{"page": p["page"], "content": p["content"]}
                                            for p in parents])
        # 4) LLM 生成（回答语言与提问一致；工单二：简洁回答控制延迟 ≤3s 目标）
        lang = r["lang"]
        system = build_system_prompt(lang, lang_override) + \
            ("回答务必简洁（150字以内），直接给出结论与页码。" if lang == "zh"
             else " Be concise (under 100 words), state the conclusion and page number directly.")
        if lang == "zh":
            prompt = (f"请根据下面的参考信息用中文回答问题。\n\n问题：{query}\n\n"
                      f"注意：参考信息中若同时出现发行人与其子公司/股东的多张同类信息表，"
                      f"请以发行人【武汉兴图新科电子股份有限公司】名下的键值信息为准。\n\n参考信息：\n{context}")
        else:
            prompt = (f"Answer the question in English based on the context below.\n\n"
                      f"Question: {query}\n\nNote: if multiple similar key-value tables exist for the issuer "
                      f"and its subsidiaries/shareholders, use the one under the issuer "
                      f"[Wuhan Xingtu Xinke Electronics Co., Ltd.].\n\nContext:\n{context}")
        t2 = time.time()
        llm = chat(messages=[{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
                   temperature=0.2, max_tokens=450)
        # 工单二容错：LLM 偶发空响应时重试一次（人工智能NLP-RAG-基于PDF文档的问答系统优化）
        if not (llm.get("content") or "").strip():
            logger.warning("LLM 空响应，重试一次")
            llm = chat(messages=[{"role": "system", "content": system},
                                 {"role": "user", "content": prompt}],
                       temperature=0.2, max_tokens=450)
        llm_ms = (time.time() - t2) * 1000
        references = [{"page": x.get("page"), "chunk_id": x.get("chunk_id"),
                       "score": x.get("score"), "preview": x.get("content", "")[:100]}
                      for x in r["results"]]
        out = {"mode": "rag_optimized", "answer": llm["content"], "references": references,
               "latency_ms": round((time.time() - t0) * 1000, 1), "cache_hit": False,
               "token_usage": llm.get("token_usage", {}), "lang": lang,
               "translated": r["translated"], "search_query": r["search_query"],
               "breakdown": {"retrieve_ms": round(retrieve_ms, 1), "llm_ms": round(llm_ms, 1)}}
        # 5) 写语义缓存（空答案绝不入缓存，防止脏数据复用，人工智能NLP-RAG-基于PDF文档的问答系统优化）
        if use_cache and (llm.get("content") or "").strip():
            cache.put(query, {"answer": llm["content"], "references": references,
                              "lang": lang}, lang=query_lang)
        return out

class RAGEngine:
    def __init__(self, retriever=None, top_k=5, max_context_chars=8000):
        self.retriever = retriever or Retriever(top_k=top_k)
        self.top_k = top_k; self.max_context_chars = max_context_chars
        self._optimized = OptimizedEngine(top_k=top_k)  # 工单二优化引擎（人工智能NLP-RAG-基于PDF文档的问答系统优化）

    def ask_optimized(self, query, lang_override=None, use_cache=True, top_k=None):
        """工单二优化问答：语义缓存+父子块+多语言路由（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
        return self._optimized.ask(query, lang_override=lang_override,
                                   use_cache=use_cache, top_k=top_k or self.top_k)

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
