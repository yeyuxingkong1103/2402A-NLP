# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from config import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TOP_K, RERANK_TOP
)

embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
rerank_model = CrossEncoder(RERANK_MODEL_PATH, device="cuda")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection("zhaogu")
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def retrieve(query, top_k=TOP_K, rerank_top=RERANK_TOP):
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    pairs = [[query, d] for d in docs]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    return ranked[:rerank_top]

def ask_rag(question):
    ranked = retrieve(question)
    contexts = [r[0] for r in ranked]
    pages = [r[1]["page"] for r in ranked]
    prompt = f"""你是一个招股说明书问答助手。请严格根据以下资料回答问题，不要编造。
如果资料中没有答案，请回答“根据招股说明书内容无法确定”。

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

if __name__ == "__main__":
    q = "武汉兴图新科电子股份有限公司法定代表人是谁？"
    ans, ctx = ask_rag(q)
    print("RAG:", ans)
    print("引用页码:", [c[0] for c in ctx])