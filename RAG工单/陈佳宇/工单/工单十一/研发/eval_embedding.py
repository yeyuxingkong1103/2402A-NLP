# eval_embedding.py 工单11 embedding评估脚本 Recall@k
# 工单编号：人工智能NLP‑RAG 项目‑Embedding 模型微调任务
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_eval_qa(filepath):
    with open(filepath,"r",encoding="utf-8") as f:
        return json.load(f)

def evaluate_embedding(model_path_or_name, eval_json_path, corpus_texts):
    """
    :param model_path_or_name: 模型名字/本地模型路径
    :param eval_json_path: 评估qa文件路径
    :param corpus_texts: 全部文档片段库
    :return: dict 指标 recall@1 recall@3
    """
    model = SentenceTransformer(model_path_or_name)
    eval_qa = load_eval_qa(eval_json_path)
    corpus_emb = model.encode(corpus_texts, show_progress_bar=False)

    recall_1 = 0.0
    recall_3 = 0.0
    total = len(eval_qa)

    for item in eval_qa:
        q = item["q"]
        gold_pos = item["pos"]
        q_emb = model.encode(q, show_progress_bar=False)
        sims = cosine_similarity([q_emb], corpus_emb)[0]
        top3_idx = np.argsort(-sims)[:3]
        top3_text = [corpus_texts[i] for i in top3_idx]
        if gold_pos == top3_text[0]:
            recall_1 += 1
        if gold_pos in top3_text:
            recall_3 +=1

    res = {
        "recall@1": round(recall_1 / total,4),
        "recall@3": round(recall_3 / total,4),
        "total_samples": total
    }
    print("====评估结果====")
    print(json.dumps(res,ensure_ascii=False,indent=2))
    return res

if __name__ == "__main__":
    corpus = [
        "武汉力源信息技术股份有限公司，是国内半导体元器件分销龙头企业。",
        "公司主营产品：半导体芯片、电子元器件、嵌入式模块。",
        "2024年公司营业收入26.8亿元，同比增长12.3%。",
        "研发部主要负责芯片方案设计，现有员工126人。",
        "公司下设部门：销售部、研发部、财务部、人力资源部。",
        "销售部包含两大业务部门：大客户销售部、中小客户销售部。",
        "大客户销售部一共设置4个销售处，分别是：华北销售处、华东销售处、华南销售处、西南销售处。",
        "中小客户销售部下辖：线上渠道组、线下渠道组。",
        "主要竞争对手：中电港、文晔科技。",
        "汽车电子业务是公司未来重点发展方向，计划加大研发投入，拓展国内车企客户。",
        "主要应用行业：工业控制、汽车电子、消费电子。",
        "公司募投项目包括：半导体测试中心建设项目、研发中心升级项目。"
    ]
    # 测试原始基线模型
    evaluate_embedding("/mnt/d/models/bge-base-zh-v1.5", "./data/eval_qa.json", corpus)
