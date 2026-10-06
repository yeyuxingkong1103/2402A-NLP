# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
RAG主流程模块：检索相关文本块 + 调用LLM生成回答
同时支持 "仅LLM回答" 用于对比分析
"""
import time
from openai import OpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, TOP_K
)

# 初始化 DeepSeek 客户端（兼容 OpenAI SDK）
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# RAG 系统提示词：要求LLM基于检索到的上下文回答
RAG_SYSTEM_PROMPT = """你是一个专业的金融文档问答助手。请根据用户提供的【参考资料】回答问题。

要求：
1. 只使用【参考资料】中的信息来回答，不要编造资料中没有的内容。
2. 如果参考资料中没有相关信息，请明确回答"根据现有资料无法找到相关信息"。
3. 回答要准确、简洁，尽量直接给出答案。
4. 对于数字、比例、金额等关键信息，请直接引用资料中的原文表述。
5. 回答使用中文。"""

# 纯LLM提示词（无参考资料，用于对比）
LLM_ONLY_SYSTEM_PROMPT = """你是一个专业的金融问答助手。请根据你已有的知识回答用户的问题。

要求：
1. 如果是关于特定公司的具体数据（如收入、注册资本等），而你不确定准确信息，请说明"不确定，建议查阅官方文件"。
2. 回答要简洁明了。
3. 回答使用中文。"""


def build_rag_prompt(question, contexts):
    """构建 RAG 的用户提示词，将检索到的上下文拼入"""
    context_str = "\n\n".join([f"【资料{i+1}】\n{c}" for i, (c, _) in enumerate(contexts)])
    user_prompt = f"""【参考资料】
{context_str}

【问题】
{question}

请基于以上参考资料回答问题。"""
    return user_prompt


def call_llm(system_prompt, user_prompt):
    """调用 LLM，返回生成的文本"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content


def rag_answer(question, vector_store, top_k=TOP_K):
    """
    RAG 回答：检索 + 生成
    返回: (answer, contexts, elapsed_time)
    """
    start = time.time()

    # 1. 检索相关文本块
    contexts = vector_store.search(question, top_k=top_k)

    # 2. 构建提示词并调用LLM
    user_prompt = build_rag_prompt(question, contexts)
    answer = call_llm(RAG_SYSTEM_PROMPT, user_prompt)

    elapsed = time.time() - start
    return answer, contexts, elapsed


def llm_only_answer(question):
    """
    纯 LLM 回答（无检索），用于对比分析
    返回: (answer, elapsed_time)
    """
    start = time.time()
    answer = call_llm(LLM_ONLY_SYSTEM_PROMPT, question)
    elapsed = time.time() - start
    return answer, elapsed


if __name__ == "__main__":
    from pdf_parser import build_chunks, load_chunks
    from vector_store import VectorStore
    import os
    from config import CHUNKS_FILE

    if os.path.exists(CHUNKS_FILE):
        chunks = load_chunks()
    else:
        chunks = build_chunks()

    store = VectorStore(chunks)

    q = "武汉兴图新科电子股份有限公司注册资本是多少？"
    print(f"问题: {q}\n")

    ans, ctxs, t = rag_answer(q, store)
    print(f"[RAG回答] (耗时{t:.2f}s)")
    print(ans)
    print(f"\n[检索到{len(ctxs)}个相关片段]")
    for i, (c, s) in enumerate(ctxs):
        print(f"  片段{i+1} (相似度{s:.4f}): {c[:100]}...")

    ans2, t2 = llm_only_answer(q)
    print(f"\n[纯LLM回答] (耗时{t2:.2f}s)")
    print(ans2)
