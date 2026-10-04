# 工单编号：人工智能 NLP-RAG-功能测试及评估
import json
import jieba
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from rank_bm25 import BM25Okapi
from config_v7 import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K, RERANK_TOP, CHUNK_FILE
)

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
rerank_model = CrossEncoder(RERANK_MODEL_PATH, device="cuda")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection("zhaogu_v7")
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)["chunks"]
corpus = [c["text"] for c in all_chunks]
bm25 = BM25Okapi([list(jieba.cut(t)) for t in corpus])

# 所有 PDF source 列表
ALL_SOURCES = list({c["source"] for c in all_chunks})


# ---------- 查询扩展 ----------
def expand_query(q):
    expansions = []
    if "保费收入" in q:
        expansions += ["保险业务收入", "已赚保费", "规模保费"]
    if "归母净利润" in q or "净利润" in q:
        expansions += ["归属于母公司股东的净利润", "净利润"]
    if "不良贷款率" in q:
        expansions += ["不良率", "资产质量"]
    if "拨备覆盖率" in q:
        expansions += ["减值准备", "贷款损失准备"]
    if "董事长致辞" in q or "致辞" in q:
        expansions += ["董事长报告", "致股东", "管理层讨论"]
    if "创新商业模式" in q:
        expansions += ["业务模式", "创新业务", "场景"]
    return q + " " + " ".join(expansions)


# ---------- 判断问题类型 ----------
def is_multi_doc(q):
    keys = ["这些", "共同", "差异", "对比", "分别", "各家", "所有"]
    return any(k in q for k in keys)


def is_early_section(q):
    keys = ["董事长", "致辞", "致股东", "管理层讨论"]
    return any(k in q for k in keys)


def is_table_query(q):
    keys = ["净利润", "保费收入", "不良贷款率", "拨备覆盖率",
            "营业收入", "投资收益", "手续费", "业务收入"]
    return any(k in q for k in keys)


# ---------- 向量召回 ----------
def vector_recall(query, source=None, top_k=TOP_K):
    where = {"source": source} if source else None
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k, where=where)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [(d, m) for d, m in zip(docs, metas)]


# ---------- BM25 召回 ----------
def bm25_recall(query, source=None, top_k=TOP_K):
    scores = bm25.get_scores(list(jieba.cut(query)))
    idx = np.argsort(scores)[::-1]
    result = []
    for i in idx:
        if source and all_chunks[i]["source"] != source:
            continue
        result.append((corpus[i], all_chunks[i]))
        if len(result) >= top_k:
            break
    return result


# ---------- 单路混合检索 ----------
def hybrid_retrieve(query, source=None, top_k=TOP_K, rerank_top=RERANK_TOP):
    vec = vector_recall(query, source=source, top_k=top_k)
    bm = bm25_recall(query, source=source, top_k=top_k)
    merged = {}
    for d, m in vec + bm:
        merged.setdefault(d, m)
    docs = list(merged.keys())
    metas = [merged[d] for d in docs]
    pairs = [[query, d] for d in docs]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    return ranked[:rerank_top]


# ---------- 多文档检索 ----------
def multi_doc_retrieve(query, rerank_top=RERANK_TOP):
    candidates = []
    per_doc = 3
    for src in ALL_SOURCES:
        sub = hybrid_retrieve(query, source=src, top_k=10, rerank_top=per_doc)
        candidates.extend(sub)
    docs = [c[0] for c in candidates]
    metas = [c[1] for c in candidates]
    pairs = [[query, d] for d in docs]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    return ranked[:rerank_top]


# ---------- 章节过滤（早期页优先） ----------
def early_section_filter(ranked, max_page=40):
    """把前 max_page 页的片段提到前面"""
    early = [r for r in ranked if r[1]["page"] <= max_page]
    late = [r for r in ranked if r[1]["page"] > max_page]
    return (early + late)[:RERANK_TOP]


# ---------- 表格优先 ----------
def table_boost(ranked, query):
    if not is_table_query(query):
        return ranked
    scored = []
    for d, m, s in ranked:
        s2 = s + (0.3 if m["type"] == "table" else 0)
        scored.append((d, m, s2))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:RERANK_TOP]


# ---------- 主检索 ----------
def retrieve(query):
    expanded = expand_query(query)

    if is_multi_doc(query):
        ranked = multi_doc_retrieve(expanded)
    else:
        ranked = hybrid_retrieve(expanded)

    if is_early_section(query):
        ranked = early_section_filter(ranked)

    ranked = table_boost(ranked, query)
    return ranked


# ---------- 生成 ----------
def ask_rag(question):
    ranked = retrieve(question)
    contexts = [r[0] for r in ranked]
    metas = [r[1] for r in ranked]
    prompt = f"""你是金融年报问答助手。请严格根据以下资料回答问题，不要编造。
如资料中确实没有答案，请回答“根据文档内容无法确定”。
如果资料中有部分相关数据，请尽量给出，并说明数据来源。

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