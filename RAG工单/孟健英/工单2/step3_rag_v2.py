# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
import json
import jieba
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from rank_bm25 import BM25Okapi
from config_v2 import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K, RERANK_TOP, CHUNK_FILE
)

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
rerank_model = CrossEncoder(RERANK_MODEL_PATH, device="cuda")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection("zhaogu_v2")
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)
corpus = [c["text"] for c in all_chunks]
tokenized = [list(jieba.cut(t)) for t in corpus]
bm25 = BM25Okapi(tokenized)


def understand_query(question):
    """Query 理解：意图识别 + 查询扩展"""
    intent = "事实查询"
    if any(k in question for k in ["多少", "比重", "金额"]):
        intent = "数值查询"
    elif any(k in question for k in ["谁", "法定代表人"]):
        intent = "实体查询"
    elif any(k in question for k in ["哪些", "包括"]):
        intent = "列表查询"

    # 查询扩展
    expansions = []
    if "军用领域" in question:
        expansions += ["直接军方", "间接军方", "军用收入", "军方收入"]
    if "收入" in question:
        expansions += ["营业收入", "主营业务收入"]
    if "上游" in question:
        expansions += ["上游行业", "供应商"]
    if "下游" in question:
        expansions += ["下游行业", "客户"]

    expanded = question + " " + " ".join(expansions)
    return {"intent": intent, "original": question, "expanded": expanded}


def hybrid_retrieve(query, top_k=TOP_K, rerank_top=RERANK_TOP):
    """向量 + BM25 混合检索 + 重排"""
    # 向量检索
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    vec_res = collection.query(query_embeddings=q_emb, n_results=top_k)
    vec_docs = vec_res["documents"][0]
    vec_pages = [m["page"] for m in vec_res["metadatas"][0]]

    # BM25 检索
    bm_scores = bm25.get_scores(list(jieba.cut(query)))
    bm_idx = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:top_k]
    bm_docs = [corpus[i] for i in bm_idx]
    bm_pages = [all_chunks[i]["page"] for i in bm_idx]

    # 合并去重
    merged = {}
    for d, p in zip(vec_docs + bm_docs, vec_pages + bm_pages):
        if d not in merged:
            merged[d] = p
    docs = list(merged.keys())
    pages = [merged[d] for d in docs]

    # 重排
    pairs = [[query, d] for d in docs]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(docs, pages, scores), key=lambda x: x[2], reverse=True)
    return ranked[:rerank_top]


def ask_rag(question):
    qu = understand_query(question)
    ranked = hybrid_retrieve(qu["expanded"])
    contexts = [r[0] for r in ranked]
    pages = [r[1] for r in ranked]
    prompt = f"""你是招股说明书问答助手。请严格根据以下资料回答问题，不要编造。
如资料中没有答案，请回答“根据招股说明书内容无法确定”。
问题意图：{qu['intent']}

资料：
{chr(10).join([f"[第{p}页] {c}" for p, c in zip(pages, contexts)])}

问题：{question}
回答："""
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content, list(zip(pages, contexts))


def ask_llm(question):
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0
    )
    return resp.choices[0].message.content