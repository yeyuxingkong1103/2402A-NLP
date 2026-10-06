# ============================================================
# 工单编号：人工智能NLP-RAG-Query理解优化任务
# 项目名称：PDF文档的Query理解优化
# 文件：test_multiturn.py
# 说明：工单 05 多轮对话测试（复现工单示例 5 轮对话）
# ============================================================
from rag_engine import RAGEngine

engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/2_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/bge-reranker-base",
)

# 工单 05 示例 5 轮对话
dialog = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]

history = []
for i, q in enumerate(dialog, 1):
    print("=" * 60)
    print(f"【第 {i} 轮】用户：{q}")
    answer, _ = engine.ask(q, history=history if history else None)
    print(f"助手：{answer}")
    history.append((q, answer))