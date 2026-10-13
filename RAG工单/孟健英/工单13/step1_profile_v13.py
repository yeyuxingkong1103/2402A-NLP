# 工单编号：人工智能 NLP-RAG-RAG 性能瓶颈识别与优化
import json
import time
import cProfile
import pstats
import io
import chromadb
import jieba
import networkx as nx
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from rank_bm25 import BM25Okapi
from config_v13 import (
    CHROMA_DIR, EMBED_MODEL_PATH, RERANK_MODEL_PATH, CHUNK_FILE, GRAPH_FILE,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    QUESTION_FILE, PROFILE_FILE
)

print("加载模型...")
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
    G.add_node(n["name"])
for e in gdata["edges"]:
    G.add_edge(e["head"], e["tail"])
entity_map = {k: set(v) for k, v in gdata["entity_map"].items()}


def timed_stage(name, fn, *args, **kwargs):
    """计时单个阶段"""
    t0 = time.time()
    result = fn(*args, **kwargs)
    dt = time.time() - t0
    return result, dt


def stage_embed(query):
    return embed_model.encode([query], normalize_embeddings=True).tolist()


def stage_vector(query_emb, top_k=20):
    return collection.query(query_embeddings=query_emb, n_results=top_k)


def stage_bm25(query, top_k=20):
    scores = bm25.get_scores(list(jieba.cut(query)))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]


def stage_graph(query):
    entities = [n for n in G.nodes if len(n) >= 2 and n in query]
    chunk_ids = set()
    for e in entities:
        chunk_ids.update(entity_map.get(e, set()))
    return list(chunk_ids)[:20]


def stage_rerank(query, docs):
    pairs = [[query, d] for d in docs]
    scores = rerank_model.predict(pairs)
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:5]


def stage_llm(query, contexts):
    prompt = f"""请根据资料回答问题：{query}
资料：{chr(10).join(contexts)}
回答："""
    t0 = time.time()
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content, time.time() - t0


if __name__ == "__main__":
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        print(f"\n[{q['id']}] {q['question']}")
        timings = {}

        _, timings["embed"] = timed_stage("embed", lambda: (emb := stage_embed(q["question"]), emb)[1])
        emb = stage_embed(q["question"])

        vec_res, timings["vector"] = timed_stage("vector", stage_vector, emb)
        bm_idx, timings["bm25"] = timed_stage("bm25", stage_bm25, q["question"])
        graph_ids, timings["graph"] = timed_stage("graph", stage_graph, q["question"])

        # 合并
        merged = {}
        for d, m in zip(vec_res["documents"][0], vec_res["metadatas"][0]):
            merged.setdefault(d, m)
        for i in bm_idx:
            merged.setdefault(corpus[i], all_chunks[i])
        for i in graph_ids:
            if i < len(all_chunks):
                merged.setdefault(all_chunks[i]["text"], all_chunks[i])

        docs = list(merged.keys())[:30]
        ranked, timings["rerank"] = timed_stage("rerank", stage_rerank, q["question"], docs)
        contexts = [r[0] for r in ranked]

        answer, timings["llm"] = stage_llm(q["question"], contexts)
        timings["total"] = sum(timings.values())

        print(f"  embed: {timings['embed']:.3f}s")
        print(f"  vector: {timings['vector']:.3f}s")
        print(f"  bm25: {timings['bm25']:.3f}s")
        print(f"  graph: {timings['graph']:.3f}s")
        print(f"  rerank: {timings['rerank']:.3f}s")
        print(f"  llm: {timings['llm']:.3f}s  ← 最大")
        print(f"  TOTAL: {timings['total']:.3f}s")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "timings": {k: round(v, 3) for k, v in timings.items()}
        })

    # 汇总
    avg = {}
    for key in ["embed", "vector", "bm25", "graph", "rerank", "llm", "total"]:
        avg[key] = round(sum(r["timings"][key] for r in results) / len(results), 3)

    print("\n===== 平均耗时 =====")
    for k, v in avg.items():
        print(f"  {k}: {v}s")

    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump({"details": results, "average": avg}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {PROFILE_FILE}")