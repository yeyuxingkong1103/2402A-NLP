# ============================================================
# 工单编号：人工智能NLP-RAG-RAG性能瓶颈识别与优化
# 文件：profile_rag_final.py
# 说明：极限优化版（max_new_tokens=128，1块上下文，300字截断）
# ============================================================
import time
from rag_engine import RAGEngine

engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/ccf_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/models/BAAI--bge-reranker-base/snapshots/master",
)

QUESTIONS = [
    "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？",
    "招商银行2019年年报中提到的核心竞争力是什么？",
    "中信证券2020年年报中的主要业务收入构成是什么？",
    "中国人寿2020年年报中的投资策略是什么？",
    "中国太保2021年年报中的保险业务发展重点是什么？",
]

def optimized_ask(q):
    stages = {}
    t0 = time.time()
    search_query = q if len(q) <= 40 else engine._rewrite_query(q)
    stages["query_rewrite"] = time.time() - t0

    t0 = time.time()
    candidates = engine._retrieve(search_query, mode="hybrid", weight=0.5, k=10)
    stages["retrieval"] = time.time() - t0

    seen = set()
    unique = []
    for d in candidates:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(d)

    t0 = time.time()
    # 极限优化：只留 1 块
    top_docs = engine._rerank(search_query, unique, top_k=1, method="bge")
    stages["rerank"] = time.time() - t0

    t0 = time.time()
    # 截断 300 字，精简 prompt
    context = top_docs[0].page_content[:300] if top_docs else ""
    prompt = f"根据上下文回答问题。\n上下文：{context}\n问题：{q}\n回答："
    # 极限优化：max_new_tokens 降到 128
    answer = engine._generate(prompt, max_new_tokens=128)
    stages["generation"] = time.time() - t0

    stages["total"] = sum(stages.values())
    return stages, answer[:100]

print("=" * 70)
print("极限优化后 RAG 性能分析")
print("=" * 70)
print(f"{'问题':<30} {'Q改写':>8} {'检索':>8} {'精排':>8} {'生成':>8} {'总计':>8}")
print("-" * 70)

all_stages = []
for q in QUESTIONS:
    stages, _ = optimized_ask(q)
    all_stages.append(stages)
    print(f"{q[:28]:<30} {stages['query_rewrite']:>7.2f}s {stages['retrieval']:>7.2f}s "
          f"{stages['rerank']:>7.2f}s {stages['generation']:>7.2f}s {stages['total']:>7.2f}s")

print("-" * 70)
avg = {k: sum(s[k] for s in all_stages) / len(all_stages) for k in all_stages[0]}
print(f"{'平均':<30} {avg['query_rewrite']:>7.2f}s {avg['retrieval']:>7.2f}s "
      f"{avg['rerank']:>7.2f}s {avg['generation']:>7.2f}s {avg['total']:>7.2f}s")
print("=" * 70)

print(f"\n优化前平均总耗时：10.83s")
print(f"第一次优化后：5.42s")
print(f"第二次优化后：{avg['total']:.2f}s")
print(f"累计提升：{(10.83 - avg['total']) / 10.83 * 100:.1f}%")
