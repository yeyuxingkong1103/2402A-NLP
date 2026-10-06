# ============================================================
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件：batch_test_03.py
# 说明：14 题批量测试（4 题力源 + 10 题兴图）
# ============================================================
import time
from rag_engine import RAGEngine

engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/2_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/bge-reranker-base",
)

questions = [
    # 力源（4题，答案在表格中）
    "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
    "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
    "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
    "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
    # 兴图（10题，沿用 01/02 工单）
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
    "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
    "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
]

for i, q in enumerate(questions, 1):
    t0 = time.time()
    answer, docs = engine.ask(q)
    elapsed = time.time() - t0
    print("=" * 60)
    print(f"[{i}/14] 问题：{q}")
    print(f"耗时：{elapsed:.1f} 秒")
    print(f"回答：{answer}")
    print()

print("批量测试完成")