# 工单编号：人工智能 NLP-RAG-RAG 性能瓶颈识别与优化
import json
import time
import hashlib
import chromadb
import jieba
import networkx as nx
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from rank_bm25 import BM25Okapi
from config_v13 import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH, CHUNK_FILE, GRAPH_FILE,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    QUESTION_FILE, COMPARE_FILE
)

# 优化点 1：嵌入和重排用 GPU + FP16
embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
embed_model.half()
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
    G.add_node(n["name"])
for e in gdata["edges"]:
    G.add_edge(e["head"], e["tail"])
entity_map = {k: set(v) for k, v in gdata["entity_map"].items()}

# 优化点 2：嵌入缓存
embed_cache = {}
# 优化点 3：答案缓存
answer_cache = {}


def get_embedding(query):
    if query in embed_cache:
        return embed_cache[query]
    emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    embed_cache[query] = emb
    return emb


def optimized_rag(question):
    t0 = time.time()

    # 缓存命中直接返回
    if question in answer_cache:
        return answer_cache[question], 0.0

    # 1. 嵌入（GPU + 缓存）
    q_emb = get_embedding(question)
    t1 = time.time()

    # 2. 向量召回
    vec_res = collection.query(query_embeddings=q_emb, n_results=20)
    # 3. BM25
    bm_scores = bm25.get_scores(list(jieba.cut(question)))
    bm_idx = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:20]
    # 4. 图扩展
    entities = [n for n in G.nodes if len(n) >= 2 and n in question]
    chunk_ids = set()
    for e in entities:
        chunk_ids.update(entity_map.get(e, set()))

    merged = {}
    for d, m in zip(vec_res["documents"][0], vec_res["metadatas"][0]):
        merged.setdefault(d, m)
    for i in bm_idx:
        merged.setdefault(corpus[i], all_chunks[i])
    for i in chunk_ids:
        if i < len(all_chunks):
            merged.setdefault(all_chunks[i]["text"], all_chunks[i])

    docs = list(merged.keys())[:30]
    t2 = time.time()

    # 5. 重排
    pairs = [[question, d] for d in docs]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:5]
    contexts = [r[0] for r in ranked]
    t3 = time.time()

    # 6. LLM 生成（优化：限制 max_tokens）
    prompt = f"""请根据资料回答问题。资料：{chr(10).join(contexts[:3])}
问题：{question}
回答（100字内）："""
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200   # 限制输出长度
    )
    answer = resp.choices[0].message.content
    t4 = time.time()

    answer_cache[question] = answer

    return answer, {
        "embed": round(t1 - t0, 3),
        "retrieve": round(t2 - t1, 3),
        "rerank": round(t3 - t2, 3),
        "llm": round(t4 - t3, 3),
        "total": round(t4 - t0, 3)
    }


if __name__ == "__main__":
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        answer, timing = optimized_rag(q["question"])
        print(f"[{q['id']}] total={timing['total'] if isinstance(timing, dict) else timing}s")
        if isinstance(timing, dict):
            print(f"  embed={timing['embed']} retrieve={timing['retrieve']} rerank={timing['rerank']} llm={timing['llm']}")
        results.append({"id": q["id"], "question": q["question"], "answer": answer, "timing": timing})

    # 汇总
    if results and isinstance(results[0]["timing"], dict):
        avg = {k: round(sum(r["timing"][k] for r in results) / len(results), 3)
               for k in ["embed", "retrieve", "rerank", "llm", "total"]}
        print("\n===== 优化后平均耗时 =====")
        for k, v in avg.items():
            print(f"  {k}: {v}s")

    with open(COMPARE_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": results, "average": avg if isinstance(results[0]["timing"], dict) else {}}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {COMPARE_FILE}")