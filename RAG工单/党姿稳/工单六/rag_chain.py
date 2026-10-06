# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
RAG主流程：检索（混合检索） + 调用 DeepSeek 生成回答。
"""
import time
from openai import OpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, TOP_K, RECALL_TOP_N,
)
from reranker import LLMReranker

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
_reranker = None


def get_reranker():
    """重排器单例"""
    global _reranker
    if _reranker is None:
        _reranker = LLMReranker()
    return _reranker

RAG_SYSTEM_PROMPT = """你是一个专业的金融文档问答助手。请严格根据用户提供的【参考资料】回答问题。

要求：
1. 只使用【参考资料】中的信息，不要编造资料中没有的内容。
2. 若资料中没有答案，明确回答"根据现有资料无法找到相关信息"。
3. 回答准确、简洁，直接给出答案；数字、比例、金额请引用原文表述。
4. 使用中文回答。"""


def _ctx_text(chunk):
    """兼容 chunk 为 dict（含页码）或纯字符串两种形式"""
    return chunk["text"] if isinstance(chunk, dict) else chunk


def build_rag_prompt(question, contexts):
    parts = []
    for i, (c, _s) in enumerate(contexts, 1):
        page = c.get("page") if isinstance(c, dict) else None
        tag = f"【资料{i}·第{page}页】" if page else f"【资料{i}】"
        parts.append(f"{tag}\n{_ctx_text(c)}")
    context_str = "\n\n".join(parts)
    return f"""【参考资料】
{context_str}

【问题】
{question}

请基于以上参考资料回答问题。"""


def call_llm(system_prompt, user_prompt):
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content


def retrieve_contexts(question, retriever, top_k=TOP_K, optimize=True, use_rerank=True):
    """
    统一的检索入口：
    optimize=True  -> 优化后：混合召回 + LLM重排
    optimize=False -> 工单一基线：仅向量检索
    返回 [(chunk_dict, score), ...]
    """
    if not optimize:
        return retriever.vector_only_search(question, top_k=top_k)
    cands = retriever.recall(question, max(RECALL_TOP_N, top_k))
    if use_rerank:
        return get_reranker().rerank(question, cands, top_k)
    return cands[:top_k]


def rag_answer(question, retriever, top_k=TOP_K, optimize=True, use_rerank=True):
    """RAG 回答，返回: (answer, contexts, elapsed)"""
    start = time.time()
    contexts = retrieve_contexts(question, retriever, top_k, optimize, use_rerank)
    answer = call_llm(RAG_SYSTEM_PROMPT, build_rag_prompt(question, contexts))
    return answer, contexts, time.time() - start


def judge_contexts(question, contexts):
    """
    LLM 裁判：判断检索到的资料是否足以回答问题（用于计算检索准确率）。
    返回 1（足够） / 0（不足）。
    """
    context_str = "\n\n".join(_ctx_text(c) for c, _ in contexts)
    prompt = f"""【问题】
{question}

【检索到的资料】
{context_str}

请判断：上述资料是否包含回答该问题所必需的关键信息？
只回答一个数字：能回答输出 1，不能输出 0。"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4,
    )
    txt = (resp.choices[0].message.content or "").strip()
    return 1 if txt.startswith("1") else 0


if __name__ == "__main__":
    import os
    from pdf_parser import build_chunks, load_chunks
    from retriever import Retriever
    from config import CHUNKS_FILE

    cs = load_chunks() if os.path.exists(CHUNKS_FILE) else build_chunks()
    r = Retriever(cs)
    q = "武汉兴图新科电子股份有限公司法定代表人是谁？"
    ans, ctxs, t = rag_answer(q, r, optimize=True)
    print(f"[优化版回答] ({t:.2f}s)\n{ans}")
    ans0, ctxs0, t0 = rag_answer(q, r, optimize=False)
    print(f"\n[基线回答] ({t0:.2f}s)\n{ans0}")
