# 工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
import json
import jieba
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from rank_bm25 import BM25Okapi
from config_v3 import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K, RERANK_TOP, CHUNK_FILE
)

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
rerank_model = CrossEncoder(RERANK_MODEL_PATH, device="cuda")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection("zhaogu_v3")
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)
corpus = [c["text"] for c in all_chunks]
tokenized = [list(jieba.cut(t)) for t in corpus]
bm25 = BM25Okapi(tokenized)


def route_source(question):
    """按公司名路由到对应 PDF"""
    if "力源" in question:
        return "liyuan"
    if "兴图" in question:
        return "xingtu"
    return None


def understand_query(question):
    """Query 理解：意图识别 + 查询扩展"""
    intent = "事实查询"
    if any(k in question for k in ["多少", "比重", "比例", "股数", "金额"]):
        intent = "数值查询"
    elif any(k in question for k in ["谁", "法定代表人"]):
        intent = "实体查询"
    elif any(k in question for k in ["哪些", "包括"]):
        intent = "列表查询"

    expansions = []
    if "军用领域" in question:
        expansions += ["直接军方", "间接军方", "军用收入"]
    if "收入" in question:
        expansions += ["营业收入", "主营业务收入"]
    if "发行股数" in question or "总股本" in question:
        expansions += ["发行股数", "总股本", "发行后总股本"]
    if "募集资金" in question or "募投" in question:
        expansions += ["募集资金", "募投项目", "投资项目"]
    if "关联方" in question:
        expansions += ["关联方", "控股股东", "持股比例"]

    expanded = question + " " + " ".join(expansions)
    return {"intent": intent, "original": question, "expanded": expanded}


def hybrid_retrieve(query, source=None, top_k=TOP_K, rerank_top=RERANK_TOP):
    """向量 + BM25 混合检索 + 表格优先 + 重排"""
    where = {"source": source} if source else None

    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    vec_res = collection.query(query_embeddings=q_emb, n_results=top_k, where=where)
    vec_docs = vec_res["documents"][0]
    vec_meta = vec_res["metadatas"][0]

    bm_scores = bm25.get_scores(list(jieba.cut(query)))
    bm_idx = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)
    if source:
        bm_idx = [i for i in bm_idx if all_chunks[i]["source"] == source]
    bm_idx = bm_idx[:top_k]
    bm_docs = [corpus[i] for i in bm_idx]
    bm_meta = [all_chunks[i] for i in bm_idx]

    merged = {}
    for d, m in zip(vec_docs + bm_docs, vec_meta + bm_meta):
        if d not in merged:
            merged[d] = m
    docs = list(merged.keys())
    metas = [merged[d] for d in docs]

    table_kw = ["股数", "比例", "关联方", "募投", "项目", "持股"]
    boost_table = any(k in query for k in table_kw)

    pairs = [[query, d] for d in docs]
    scores = rerank_model.predict(pairs)
    scored = []
    for d, m, s in zip(docs, metas, scores):
        s2 = s + (0.2 if boost_table and m["type"] == "table" else 0)
        scored.append((d, m, s2))
    ranked = sorted(scored, key=lambda x: x[2], reverse=True)
    return ranked[:rerank_top]


def ask_rag(question):
    qu = understand_query(question)
    src = route_source(question)
    ranked = hybrid_retrieve(qu["expanded"], source=src)
    contexts = [r[0] for r in ranked]
    metas = [r[1] for r in ranked]
    prompt = f"""你是招股说明书问答助手。请严格根据以下资料回答问题，不要编造。
如资料中没有答案，请回答“根据招股说明书内容无法确定”。
问题意图：{qu['intent']}

资料：
{chr(10).join([f"[{m['source']} 第{m['page']}页 {m['type']}] {c}" for m, c in zip(metas, contexts)])}

问题：{question}
回答："""
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content, list(zip(metas, contexts))


def ask_llm(question):
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0
    )
    return resp.choices[0].message.content