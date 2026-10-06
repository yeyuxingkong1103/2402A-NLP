# 工单编号：人工智能 NLP-RAG-Query 理解优化任务
import json
import jieba
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from rank_bm25 import BM25Okapi
from config_v5 import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K, RERANK_TOP, CHUNK_FILE
)

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
rerank_model = CrossEncoder(RERANK_MODEL_PATH, device="cuda")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection("zhaogu_v5")
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
all_chunks = data["chunks"]
corpus = [c["text"] for c in all_chunks]
bm25 = BM25Okapi([list(jieba.cut(t)) for t in corpus])

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
        self.history.append({
            "question": question,
            "rewritten": rewritten,
            "company": company,
            "attribute": attribute
        })


def detect_company(q):
    if "兴图" in q:
        return COMPANY_XINGTU
    if "力源" in q:
        return COMPANY_LIYUAN
    return None


def detect_attribute(q):
    attrs = [
        ("法定代表人", "法定代表人"),
        ("注册资本", "注册资本"),
        ("军用领域收入", "军用领域收入"),
        ("军用领域", "军用领域收入"),
        ("技术标准", "技术标准"),
        ("科技进步一等奖", "科技进步一等奖"),
        ("补充流动资金", "补充流动资金"),
        ("组织结构", "组织结构图"),
        ("销售处", "组织结构图"),
        ("上游", "行业上游"),
        ("下游", "行业下游"),
    ]
    for k, v in attrs:
        if k in q:
            return v
    return None


def rewrite_query(question, state):
    company = detect_company(question)
    attribute = detect_attribute(question)

    # 情况 1：省略式“那XX呢？”优先处理
    is_ellipsis = ("那" in question and "呢" in question)
    if is_ellipsis and state.last_company:
        new_company = detect_company(question) or state.last_company
        attr = state.last_attribute or "法定代表人"
        rewritten = f"{new_company}的{attr}是多少？"
        return rewritten, new_company, attr

    # 情况 2：明确出现公司名，不重写
    if company:
        return question, company, attribute

    # 情况 3：指代词 + 上下文有公司
    pronouns = ["他", "她", "它", "这个公司", "该公司", "这家公司", "其"]
    if any(p in question for p in pronouns) and state.last_company:
        rewritten = question
        for p in pronouns:
            rewritten = rewritten.replace(p, state.last_company)
        return rewritten, state.last_company, attribute

    # 情况 4：LLM 兜底重写
    if state.last_company:
        prompt = f"""请把下面的问题改写成一个完整问题，补全省略和指代。
上一轮提到的公司：{state.last_company}
上一轮的问题属性：{state.last_attribute or "未知"}
当前问题：{question}
改写后（只输出改写结果，不要解释）："""
        resp = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten, state.last_company, attribute

    return question, None, attribute

def route_source(question):
    if "力源" in question:
        return "liyuan"
    if "兴图" in question:
        return "xingtu"
    return None


def is_image_query(q):
    keys = ["图", "组织结构", "组织架构", "增长率", "结构图", "销售处"]
    return any(k in q for k in keys)


def hybrid_retrieve(query, source=None, top_k=TOP_K, rerank_top=RERANK_TOP):
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

    merged = {}
    for d, m in zip(vec_docs, vec_meta):
        merged.setdefault(d, m)
    for i in bm_idx:
        merged.setdefault(corpus[i], all_chunks[i])

    docs = list(merged.keys())
    metas = [merged[d] for d in docs]

    pairs = [[query, d] for d in docs]
    scores = rerank_model.predict(pairs)

    table_kw = ["股数", "比例", "关联方", "募投", "持股"]
    boost_table = any(k in query for k in table_kw)
    boost_image = is_image_query(query)

    scored = []
    for d, m, s in zip(docs, metas, scores):
        s2 = s
        if boost_table and m["type"] == "table":
            s2 += 0.2
        if boost_image and m["type"] == "image":
            s2 += 0.3
        scored.append((d, m, s2))
    ranked = sorted(scored, key=lambda x: x[2], reverse=True)
    return ranked[:rerank_top]


def ask_rag(question, state=None):
    if state is None:
        state = DialogState()
    rewritten, company, attribute = rewrite_query(question, state)
    state.update(question, rewritten, company, attribute)

    src = route_source(rewritten)
    ranked = hybrid_retrieve(rewritten, source=src)
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