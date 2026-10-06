# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
RAG 核心：跨两份招股书的向量检索 + LLM 生成（Ollama 官方 SDK，固定端点 127.0.0.1:11434）
在工单02 基础上：检索结果带文档名/页码/块类型，提示词要求标注出处文档
"""
import re

import chromadb
import ollama

CHROMA_DIR = "chroma_db"
COLLECTIONS = ["zhaogu1_v3", "zhaogu2_v3"]  # 按文档拆分（chromadb 大集合跨进程重载有 bug）
TOP_K = 6
MIN_SCORE = 0.35
GEN_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

DOC_ALIAS = {"zhaogu1": "招股说明书1（兴图新科）", "zhaogu2": "招股说明书2（力源信息）"}

PROMPT_TMPL = """你是证券文档问答助手。规则：
1. 仅根据【检索上下文】回答【问题】，不得编造；
2. 只要上下文中出现了与问题对应的内容，就必须直接给出答案（可摘录原文数字与表述），
   禁止因为"信息不够全面/详细"而拒绝回答；
3. 上下文中确实完全没有答案时，才回答"根据文档内容未找到"；
4. 表格中的数值是权威答案，涉及金额/股数/比例时优先引用表格数据；
5. 回答末尾标注出处，格式如（招股说明书1 第X页）。

【检索上下文】
{context}

【问题】{question}

【回答】"""


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return [client.get_collection(c) for c in COLLECTIONS]


def clean_query(question: str) -> str:
    """查询清洗：去公司全称/固定前缀，避免嵌入被高频词稀释"""
    q = re.sub(r"根据?.{0,20}?招股(说明书|意向书)[，,]?", "", question)
    for name in ("武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司"):
        q = q.replace(name, "")
    q = q.replace("报告期内，", "")
    return q.strip("，, 。") or question


def embed_query(text: str):
    return ollama.embed(model=EMBED_MODEL, input=[text])["embeddings"][0]


def retrieve(col, question: str, top_k: int = TOP_K):
    """跨两份招股书集合检索，按相似度合并"""
    vec = embed_query(clean_query(question))
    hits = []
    for c in col:
        res = c.query(query_embeddings=[vec], n_results=top_k)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = round(1 - dist, 4)
            if score >= MIN_SCORE:
                hits.append({"text": doc, "doc": meta["doc"], "page": meta["page"],
                             "kind": meta.get("kind", "text"), "score": score})
    hits.sort(key=lambda h: -h["score"])
    return hits[:top_k]


def llm_generate(prompt: str) -> str:
    resp = ollama.generate(model=GEN_MODEL, prompt=prompt, stream=False,
                           options={"temperature": 0.1})
    return resp["response"].strip()


def answer(col, question: str, top_k: int = TOP_K):
    """返回 (答案, 命中块列表)"""
    hits = retrieve(col, question, top_k)
    ctx = "\n\n".join(f"[{DOC_ALIAS[h['doc']]} 第{h['page']}页{' 表格' if h['kind']=='table' else ''}] {h['text']}"
                      for h in hits)
    prompt = PROMPT_TMPL.format(context=ctx, question=clean_query(question))
    ans = llm_generate(prompt)
    return ans, hits
