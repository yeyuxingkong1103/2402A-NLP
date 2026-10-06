# ============================================================
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 项目名称：PDF文档的功能测试及评估
# 文件：test_07.py
# 说明：对 CCF 年报向量库进行 10 个问题的测试
# ============================================================
import time
import json
from rag_engine import RAGEngine

# 使用 CCF 向量库
CCF_DB = "/root/autodl-tmp/projects/RAG/ccf_vector_db"

engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path=CCF_DB,
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/bge-reranker-base",
)

# 10 个问题（前 4 个来自 sample_questions.pdf，后 6 个扩展）
QUESTIONS = [
    # 来自 sample_questions.pdf
    "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？",
    "平安银行在其2019年年报中提到的创新商业模式有哪些？",
    "平安银行的风险管理体系如何体现其应对宏观经济周期波动的能力？结合年报中的拨备覆盖率动态调整、资产质量控制，以及贷款结构优化策略，分析其应对经济下行周期的准备程度及可能的潜在压力。",
    "分析这些银行和保险公司在面对经济周期波动时的共同策略与差异化策略，重点关注其在风险管理、资本结构优化和新兴领域（如绿色金融或科技金融）投资上的表现。",
    # 扩展 6 题
    "招商银行2019年年报中提到的核心竞争力是什么？",
    "中国平安2019年年报中主要业务板块及各自的盈利情况如何？",
    "中信证券2020年年报中的主要业务收入构成是什么？",
    "中国人寿2020年年报中的投资策略是什么？",
    "中国太保2021年年报中的保险业务发展重点是什么？",
    "国泰君安2021年年报中的风险管理措施有哪些？",
]

results = []
for i, q in enumerate(QUESTIONS, 1):
    t0 = time.time()
    answer, docs = engine.ask(
        q,
        retrieval_mode="hybrid",
        hybrid_weight=0.5,
        rerank_method="bge",
    )
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"[{i}/10] 问题：{q}")
    print(f"耗时：{elapsed:.1f} 秒")
    print(f"回答：{answer}")
    print(f"\n检索来源（前 3 个块）：")
    for j, d in enumerate(docs[:3]):
        src = d.metadata.get("source", "未知") if hasattr(d, "metadata") else "未知"
        print(f"  [块{j+1}] 来源：{src} | {d.page_content[:120]}...")
    print()

    results.append({
        "id": i,
        "question": q,
        "answer": answer,
        "elapsed": round(elapsed, 1),
        "sources": [d.page_content[:300] for d in docs[:3]],
    })

# 保存结果
with open("/root/autodl-tmp/projects/RAG/test_07_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("=" * 70)
print(f"测试完成，结果已保存到 test_07_results.json")