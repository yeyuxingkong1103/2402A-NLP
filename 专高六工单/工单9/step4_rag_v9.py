# 工单编号：人工智能 NLP-RAG-Graph RAG 优化任务
import json
import time
import jieba
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from rank_bm25 import BM25Okapi
from config_v9 import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K, RERANK_TOP, CHUNK_FILE, GRAPH_FILE, GRAPH_HOPS
)

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
rerank_model = CrossEncoder(RERANK_MODEL_PATH, device="cuda")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection("zhaogu_v9")
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)["chunks"]
corpus = [c["text"] for c in all_chunks]
bm25 = BM25Okapi([list(jieba.cut(t)) for t in corpus])

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    gdata = json.load(f)
G = nx.DiGraph()
for n in gdata["nodes"]:
    G.add_node(n["name"], **{k: v for k, v in n.items() if k != "name"})
for e in gdata["edges"]:
    G.add_edge(e["head"], e["tail"], **{k: v for k, v in e.items() if k not in ("head", "tail")})
entity_map = {k: set(v) for k, v in gdata["entity_map"].items()}

ALL_SOURCES = list({c["source"] for c in all_chunks})

COMPANY_KEYS = {
    "平安银行": "平安银行",
    "中国平安": "中国平安",
    "招商银行": "招商银行",
    "邮储银行": "邮储银行",
    "中信证券": "中信证券",
    "中国人寿": "中国人寿",
    "中国太保": "中国太保",
    "招商证券": "招商证券",
    "国泰君安": "国泰君安",
}


def route_sources(query):
    matched = [v for k, v in COMPANY_KEYS.items() if k in query]
    if not matched:
        return None
    result = []
    for src in ALL_SOURCES:
        for m in matched:
            if m in src:
                result.append(src)
    return list(set(result))


def match_entities(query):
    return [name for name in G.nodes if len(name) >= 2 and name in query]


def graph_expand(query):
    entities = match_entities(query)
    chunk_ids = set()
    for name in entities:
        chunk_ids.update(entity_map.get(name, set()))
    result = []
    for i in chunk_ids:
        if i >= len(all_chunks):
            continue
        result.append((all_chunks[i]["text"], all_chunks[i]))
    return result


def vector_recall(query, sources=None, top_k=30):
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    if sources and len(sources) == 1:
        res = collection.query(query_embeddings=q_emb, n_results=top_k, where={"source": sources[0]})
        return list(zip(res["documents"][0], res["metadatas"][0]))
    res = collection.query(query_embeddings=q_emb, n_results=top_k * 2)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    if sources:
        return [(d, m) for d, m in zip(docs, metas) if m["source"] in sources][:top_k]
    return list(zip(docs, metas))[:top_k]


def graph_retrieve(query, return_timing=False):
    t0 = time.time()
    sources = route_sources(query)
    vec = vector_recall(query, sources=sources, top_k=30)
    graph = graph_expand(query)

    merged = {}
    for d, m in vec + graph:
        merged.setdefault(d, m)
    docs = list(merged.keys())
    metas = [merged[d] for d in docs]

    table_keys = ["净利润", "保费收入", "不良贷款率", "拨备覆盖率", "营业收入", "投资收益", "手续费", "已赚保费"]
    boost_table = any(k in query for k in table_keys)

    if len(docs) > 15:
        q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
        sub = collection.query(query_embeddings=q_emb, n_results=15)
        keep = set(sub["documents"][0])
        filtered = [(d, metas[i]) for i, d in enumerate(docs) if d in keep]
        if len(filtered) >= 6:
            docs = [d for d, _ in filtered]
            metas = [m for _, m in filtered]

    pairs = [[query, d] for d in docs]
    scores = rerank_model.predict(pairs)
    scored = []
    for d, m, s in zip(docs, metas, scores):
        s2 = s + (0.5 if boost_table and m["type"] == "table" else 0)
        scored.append((d, m, s2))
    ranked = sorted(scored, key=lambda x: x[2], reverse=True)

    # 固定 Top-5
    ranked = ranked[:5]

    t1 = time.time()
    if return_timing:
        return ranked, t1 - t0
    return ranked

def ask_graph_rag(question, return_timing=False):
    if return_timing:
        ranked, rt = graph_retrieve(question, return_timing=True)
    else:
        ranked = graph_retrieve(question)
        rt = None
    contexts = [r[0] for r in ranked]
    metas = [r[1] for r in ranked]
    prompt = f"""你是金融年报问答助手。请严格根据以下资料回答问题。
要求：
1. 只回答资料中明确出现的信息，不要推测
2. 资料中没有答案，直接回答“根据文档内容无法确定”
3. 答案控制在 200 字内

资料：
{chr(10).join([f"[{m['source']} 第{m['page']}页 {m['type']}] {c}" for m, c in zip(metas, contexts)])}

问题：{question}
回答："""
    t0 = time.time()
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    t1 = time.time()
    answer = resp.choices[0].message.content
    if return_timing:
        return answer, list(zip(metas, contexts)), rt + (t1 - t0)
    return answer, list(zip(metas, contexts))