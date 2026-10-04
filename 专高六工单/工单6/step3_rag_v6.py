# 工单编号：人工智能 NLP-RAG-混合检索任务
import json
import os
import re
import jieba
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from rank_bm25 import BM25Okapi
from config_v6 import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K, RERANK_TOP, CHUNK_FILE, FEEDBACK_FILE,
    DEFAULT_VECTOR_WEIGHT, DEFAULT_FULLTEXT_WEIGHT
)

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
rerank_model = CrossEncoder(RERANK_MODEL_PATH, device="cuda")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection("zhaogu_v6")
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
all_chunks = data["chunks"]
corpus = [c["text"] for c in all_chunks]
tokenized_corpus = [list(jieba.cut(t)) for t in corpus]
bm25 = BM25Okapi(tokenized_corpus)

tfidf_vec = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)), token_pattern=None)
tfidf_matrix = tfidf_vec.fit_transform(corpus)

if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        feedback = json.load(f)
else:
    feedback = {}


# ---------- 全文检索 ----------
def fulltext_search(query, top_k=TOP_K):
    bool_and = re.split(r"且|AND|and", query)
    phrases = re.findall(r'"([^"]+)"', query)
    tokens = [t.strip() for t in jieba.cut(query) if t.strip()]

    scores = bm25.get_scores(tokens)

    if len(bool_and) > 1:
        for part in bool_and:
            sub_tokens = [t for t in jieba.cut(part) if t.strip()]
            sub_scores = bm25.get_scores(sub_tokens)
            scores = scores * (sub_scores > 0)

    for phrase in phrases:
        for i, doc in enumerate(corpus):
            if phrase in doc:
                scores[i] += 2.0

    for i, doc in enumerate(corpus):
        overlap = len(set(query) & set(doc)) / max(len(set(query)), 1)
        scores[i] += 0.1 * overlap

    idx = np.argsort(scores)[::-1][:top_k]
    return [(corpus[i], all_chunks[i], float(scores[i])) for i in idx]


# ---------- 向量检索 ----------
def vector_search(query, source=None, top_k=TOP_K):
    where = {"source": source} if source else None
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k, where=where)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [(d, m, 1.0) for d, m in zip(docs, metas)]


# ---------- 重排 ----------
def rerank_with_bge(query, candidates, top_n=RERANK_TOP):
    pairs = [[query, c[0]] for c in candidates]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(c[0], c[1], float(s)) for c, s in ranked[:top_n]]


def rerank_with_tfidf(query, candidates, top_n=RERANK_TOP):
    q_vec = tfidf_vec.transform([query])
    scored = []
    for doc, meta, _ in candidates:
        d_vec = tfidf_vec.transform([doc])
        sim = (q_vec @ d_vec.T).toarray()[0][0]
        scored.append((doc, meta, float(sim)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


def rerank_with_llm(query, candidates, top_n=RERANK_TOP):
    if not candidates:
        return []
    items = "\n".join([f"{i+1}. {c[0][:200]}" for i, c in enumerate(candidates)])
    prompt = f"""请给下面每个片段与问题的相关性打分（0-10）。
问题：{query}
片段：
{items}
输出格式（每行一个分数，顺序对应）："""
    try:
        resp = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = resp.choices[0].message.content
        scores = [float(x) for x in re.findall(r"\d+\.?\d*", text)][:len(candidates)]
        if len(scores) < len(candidates):
            scores += [0.0] * (len(candidates) - len(scores))
        scored = [(c[0], c[1], s) for c, s in zip(candidates, scores)]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]
    except Exception:
        return rerank_with_bge(query, candidates, top_n)


def rerank_adaptive(query, candidates, top_n=RERANK_TOP):
    base = rerank_with_bge(query, candidates, top_n=len(candidates))
    scored = []
    for doc, meta, s in base:
        fid = f"{meta['source']}_{meta['page']}_{meta['type']}"
        fb = feedback.get(fid, 0)
        scored.append((doc, meta, s + 0.1 * fb))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


RERANKERS = {
    "bge": rerank_with_bge,
    "tfidf": rerank_with_tfidf,
    "llm": rerank_with_llm,
    "adaptive": rerank_adaptive,
}


# ---------- 混合检索 ----------
def hybrid_search(query, source=None, vector_weight=DEFAULT_VECTOR_WEIGHT,
                  fulltext_weight=DEFAULT_FULLTEXT_WEIGHT, top_k=TOP_K):
    vec = vector_search(query, source=source, top_k=top_k)
    full = fulltext_search(query, top_k=top_k)

    def norm(items):
        if not items:
            return []
        scores = [i[2] for i in items]
        mx = max(scores) or 1
        mn = min(scores)
        return [(i[0], i[1], (i[2] - mn) / (mx - mn + 1e-9)) for i in items]

    vec = norm(vec)
    full = norm(full)

    merged = {}
    for d, m, s in vec:
        merged.setdefault(d, {"meta": m, "score": 0})
        merged[d]["score"] += vector_weight * s
    for d, m, s in full:
        merged.setdefault(d, {"meta": m, "score": 0})
        merged[d]["score"] += fulltext_weight * s

    ranked = sorted(merged.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
    return [(d, v["meta"], v["score"]) for d, v in ranked]


# ---------- 主入口 ----------
def retrieve(query, mode="hybrid", reranker="bge", source=None,
             vector_weight=DEFAULT_VECTOR_WEIGHT,
             fulltext_weight=DEFAULT_FULLTEXT_WEIGHT):
    if mode == "vector":
        cands = vector_search(query, source=source, top_k=TOP_K)
    elif mode == "fulltext":
        cands = fulltext_search(query, top_k=TOP_K)
    else:
        cands = hybrid_search(query, source=source,
                              vector_weight=vector_weight,
                              fulltext_weight=fulltext_weight,
                              top_k=TOP_K)
    rerank_fn = RERANKERS.get(reranker, rerank_with_bge)
    return rerank_fn(query, cands)


# ---------- 多轮对话 + Query 重写 ----------
COMPANY_XINGTU = "武汉兴图新科电子股份有限公司"
COMPANY_LIYUAN = "武汉力源信息技术股份有限公司"


class DialogState:
    def __init__(self):
        self.last_company = None
        self.last_attribute = None
        self.history = []

    def update(self, question, rewritten, company, attribute):
        self.last_company = company or self.last_company
        self.last_attribute = attribute or self.last_attribute
        self.history.append({"question": question, "rewritten": rewritten})


def detect_company(q):
    if "兴图" in q:
        return COMPANY_XINGTU
    if "力源" in q:
        return COMPANY_LIYUAN
    return None


def detect_attribute(q):
    for k in ["法定代表人", "注册资本", "军用领域收入", "技术标准",
              "科技进步一等奖", "补充流动资金", "组织结构", "销售处", "上游", "下游"]:
        if k in q:
            return k
    return None


def rewrite_query(question, state):
    company = detect_company(question)
    attribute = detect_attribute(question)

    is_ellipsis = ("那" in question and "呢" in question)
    if is_ellipsis and state.last_company:
        new_company = detect_company(question) or state.last_company
        attr = state.last_attribute or "法定代表人"
        return f"{new_company}的{attr}是多少？", new_company, attr

    if company:
        return question, company, attribute

    pronouns = ["他", "她", "它", "这个公司", "该公司", "这家公司", "其"]
    if any(p in question for p in pronouns) and state.last_company:
        rewritten = question
        for p in pronouns:
            rewritten = rewritten.replace(p, state.last_company)
        return rewritten, state.last_company, attribute

    return question, None, attribute


def route_source(q):
    if "力源" in q:
        return "liyuan"
    if "兴图" in q:
        return "xingtu"
    return None


def ask_rag(question, state=None, mode="hybrid", reranker="bge",
            vector_weight=DEFAULT_VECTOR_WEIGHT,
            fulltext_weight=DEFAULT_FULLTEXT_WEIGHT):
    if state is None:
        state = DialogState()
    rewritten, company, attribute = rewrite_query(question, state)
    state.update(question, rewritten, company, attribute)

    src = route_source(rewritten)
    ranked = retrieve(rewritten, mode=mode, reranker=reranker,
                      source=src, vector_weight=vector_weight,
                      fulltext_weight=fulltext_weight)
    contexts = [r[0] for r in ranked]
    metas = [r[1] for r in ranked]

    prompt = f"""你是招股说明书问答助手。请严格根据以下资料回答问题，不要编造。
如资料中没有答案，请回答“根据招股说明书内容无法确定”。

资料：
{chr(10).join([f"[{m['source']} 第{m['page']}页 {m['type']}] {c}" for m, c in zip(metas, contexts)])}

问题：{rewritten}
回答："""
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content, list(zip(metas, contexts)), rewritten


def ask_llm(question):
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0
    )
    return resp.choices[0].message.content


def save_feedback(source, page, type_, score):
    fid = f"{source}_{page}_{type_}"
    feedback[fid] = feedback.get(fid, 0) + score
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)