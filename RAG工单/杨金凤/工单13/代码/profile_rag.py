# ============================================================
# 工单编号：人工智能NLP-RAG-RAG性能瓶颈识别与优化
# 文件：profile_rag.py
# 说明：分阶段计时，找出 RAG 系统的性能瓶颈
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

# 分阶段计时
def profile_ask(q):
    stages = {}

    # 阶段1: Query 改写
    t0 = time.time()
    if len(q) <= 40:
        search_query = q
    else:
        search_query = engine._rewrite_query(q)
    stages["query_rewrite"] = time.time() - t0

    # 阶段2: 检索（向量 + BM25 + RRF）
    t0 = time.time()
    candidates = engine._retrieve(search_query, mode="hybrid", weight=0.5, k=15)
    stages["retrieval"] = time.time() - t0

    # 去重
    seen = set()
    unique = []
    for d in candidates:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(d)

    # 阶段3: Reranker 精排
    t0 = time.time()
    top_docs = engine._rerank(search_query, unique, top_k=4, method="bge")
    stages["rerank"] = time.time() - t0

    # 阶段4: LLM 生成
    t0 = time.time()
    context = "\n\n".join([d.page_content for d in top_docs])
    prompt = f"""你是金融年报问答助手。请严格基于以下上下文回答问题。
上下文：
{context}
问题：{q}
回答："""
    answer = engine._generate(prompt, max_new_tokens=512)
    stages["generation"] = time.time() - t0

    stages["total"] = sum(stages.values())
    return stages, answer[:100]


print("=" * 70)
print("RAG 性能分析（每个阶段耗时）")
print("=" * 70)
print(f"{'问题':<30} {'Q改写':>8} {'检索':>8} {'精排':>8} {'生成':>8} {'总计':>8}")
print("-" * 70)

all_stages = []
for q in QUESTIONS:
    stages, _ = profile_ask(q)
    all_stages.append(stages)
    print(f"{q[:28]:<30} {stages['query_rewrite']:>7.2f}s {stages['retrieval']:>7.2f}s "
          f"{stages['rerank']:>7.2f}s {stages['generation']:>7.2f}s {stages['total']:>7.2f}s")

# 平均
print("-" * 70)
avg = {k: sum(s[k] for s in all_stages) / len(all_stages) for k in all_stages[0]}
print(f"{'平均':<30} {avg['query_rewrite']:>7.2f}s {avg['retrieval']:>7.2f}s "
      f"{avg['rerank']:>7.2f}s {avg['generation']:>7.2f}s {avg['total']:>7.2f}s")
print("=" * 70)

# 瓶颈分析
bottleneck = max(avg, key=lambda k: avg[k] if k != "total" else 0)
print(f"\n★ 最大瓶颈：{bottleneck}（{avg[bottleneck]:.2f}s，占 {avg[bottleneck]/avg['total']*100:.1f}%）")
