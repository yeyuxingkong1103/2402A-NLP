# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
优化版 RAG 核心（对比工单01的三处改进）：
  1) 检索：top_k 5 -> 6，返回相似度过滤低分块
  2) 提示词：上下文含答案时必须直接作答，禁止以"不够全面"为由拒答
  3) 引用：答案后强制附页码
LLM 与嵌入均通过本机 Ollama 官方 SDK（固定端点 127.0.0.1:11434）
"""
import chromadb
import ollama

CHROMA_DIR = "chroma_db"
COLLECTION = "zhaogu1_v2"
TOP_K = 6
MIN_SCORE = 0.35          # 低于该相似度的块不进入上下文
GEN_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

PROMPT_TMPL = """你是证券文档问答助手。规则：
1. 仅根据【检索上下文】回答【问题】，不得编造；
2. 只要上下文中出现了与问题对应的内容，就必须直接给出答案（可摘录原文数字与表述），
   禁止因为"信息不够全面/详细"而拒绝回答；
3. 上下文中确实完全没有答案时，才回答"根据文档内容未找到"；
4. 回答末尾标注所引用内容的页码，格式如（第X页）。

【检索上下文】
{context}

【问题】{question}

【回答】"""


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def embed_query(text: str):
    return ollama.embed(model=EMBED_MODEL, input=[text])["embeddings"][0]


def clean_query(question: str) -> str:
    """查询清洗：去掉公司全称/固定前缀等高频词，避免嵌入被稀释、套话块虚高"""
    import re
    q = re.sub(r"根据?.{0,20}?招股(说明书|意向书)[，,]?", "", question)
    q = q.replace("武汉兴图新科电子股份有限公司", "").replace("报告期内，", "")
    return q.strip("，, 。") or question


def retrieve(col, question: str, top_k: int = TOP_K):
    vec = embed_query(clean_query(question))
    res = col.query(query_embeddings=[vec], n_results=top_k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = round(1 - dist, 4)
        if score >= MIN_SCORE:
            hits.append({"text": doc, "page": meta["page"], "score": score})
    return hits


def llm_generate(prompt: str) -> str:
    resp = ollama.generate(model=GEN_MODEL, prompt=prompt, stream=False,
                           options={"temperature": 0.1})
    return resp["response"].strip()


def answer(col, question: str, top_k: int = TOP_K):
    """返回 (答案, 命中块列表)。LLM 同样使用清洗后的问题，避免其对'招股意向书/说明书'等措辞抠字眼拒答"""
    hits = retrieve(col, question, top_k)
    ctx = "\n\n".join(f"[第{h['page']}页] {h['text']}" for h in hits)
    prompt = PROMPT_TMPL.format(context=ctx, question=clean_query(question))
    ans = llm_generate(prompt)
    return ans, hits
