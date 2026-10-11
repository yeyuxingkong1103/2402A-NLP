# ============================================================
# 工单编号：人工智能NLP-RAG-Embedding模型微调任务
# 文件：build_finetune_data.py
# 说明：从 CCF 向量库中为 10 个问题检索正确文档，
#       构造 (query, positive) 训练对，输出 train.jsonl
# ============================================================
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ground_truth_ccf import GROUND_TRUTH_CCF

# 加载原始嵌入模型和 CCF 向量库
embeddings = HuggingFaceEmbeddings(
    model_name="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma(
    persist_directory="/root/autodl-tmp/projects/RAG/ccf_vector_db",
    embedding_function=embeddings,
)

# 对每个问题检索 top3 文档作为 positive
train_data = []
for q in GROUND_TRUTH_CCF.keys():
    docs = vectordb.similarity_search(q, k=3)
    for d in docs:
        train_data.append({
            "query": q,
            "positive": d.page_content,
        })

# 保存
with open("/root/autodl-tmp/projects/RAG/train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"训练数据生成完成：{len(train_data)} 条 (query, positive) 对")