# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
RAG 核心：向量检索 + LLM 生成
LLM 与嵌入均通过本机 Ollama 官方 SDK 调用（固定端点 127.0.0.1:11434，不接受外部 URL）
供 demo.py 复用
"""
import chromadb
import ollama

CHROMA_DIR = "chroma_db"
COLLECTION = "zhaogu1_v1"
TOP_K = 5
GEN_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

PROMPT_TMPL = """你是证券文档问答助手。请仅根据下面的检索上下文回答问题，\
不要编造；若上下文中没有答案，请回答"根据文档内容未找到"。回答末尾标注引用的页码。

【检索上下文】
{context}

【问题】{question}

【回答】"""


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def embed_query(text: str):
    return ollama.embed(model=EMBED_MODEL, input=[text])["embeddings"][0]


def retrieve(col, question: str, top_k: int = TOP_K):
    vec = embed_query(question)
    res = col.query(query_embeddings=[vec], n_results=top_k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "page": meta["page"], "score": round(1 - dist, 4)})
    return hits


def llm_generate(prompt: str) -> str:
    resp = ollama.generate(model=GEN_MODEL, prompt=prompt, stream=False,
                           options={"temperature": 0.1})
    return resp["response"].strip()


def answer(col, question: str, top_k: int = TOP_K):
    """返回 (答案, 命中块列表)"""
    hits = retrieve(col, question, top_k)
    ctx = "\n\n".join(f"[第{h['page']}页] {h['text']}" for h in hits)
    prompt = PROMPT_TMPL.format(context=ctx, question=question)
    ans = llm_generate(prompt)
    return ans, hits
