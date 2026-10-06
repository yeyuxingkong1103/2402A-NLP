# ============================================================
# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件：test_hybrid.py
# 说明：对比三种检索模式 × 三种重排算法的效果
# ============================================================
import time
from rag_engine import RAGEngine

engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/2_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/bge-reranker-base",
)

# 测试用题
q = "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"

print("=" * 60)
print(f"测试问题：{q}")
print("=" * 60)

# 3 种检索模式（重排固定用 bge）
for mode in ["vector", "fulltext", "hybrid"]:
    t0 = time.time()
    ans, docs = engine.ask(q, retrieval_mode=mode, rerank_method="bge")
    elapsed = time.time() - t0
    print(f"\n【检索模式：{mode}】耗时 {elapsed:.1f}s")
    print(f"回答：{ans[:200]}")

print("\n" + "=" * 60)
print("3 种重排算法对比（检索模式固定 hybrid）")
print("=" * 60)

for method in ["bge", "tfidf", "llm"]:
    t0 = time.time()
    ans, docs = engine.ask(q, retrieval_mode="hybrid", rerank_method=method)
    elapsed = time.time() - t0
    print(f"\n【重排算法：{method}】耗时 {elapsed:.1f}s")
    print(f"回答：{ans[:200]}")