# main_finetune.py 工单11 embedding微调主脚本
# 工单编号：人工智能NLP‑RAG 项目‑Embedding 模型微调任务
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
from eval_embedding import evaluate_embedding

# 业务文档语料
corpus_texts = [
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

def load_triple_dataset(jsonl_path):
    examples = []
    with open(jsonl_path,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            examples.append(InputExample(texts=[d["anchor"],d["positive"],d["negative"]]))
    return examples

def main():
    model_name = "/mnt/d/models/bge-base-zh-v1.5"
    train_data_path = "./data/train_triple.jsonl"
    eval_qa_path = "./data/eval_qa.json"
    out_model_dir = "./output_model"

    print("【1】加载基线预训练模型",model_name)
    model = SentenceTransformer(model_name)

    print("\n【2】微调前基线模型评估")
    base_metric = evaluate_embedding(model_name, eval_qa_path, corpus_texts)

    print("\n【3】加载训练三元组数据集")
    train_examples = load_triple_dataset(train_data_path)
    print(f"训练样本数量：{len(train_examples)}")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
    train_loss = losses.TripletLoss(model=model)

    print("\n【4】开始CPU微调训练")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=0,
        output_path=out_model_dir,
        show_progress_bar=True
    )
    print(f"训练完成，微调模型保存到 {out_model_dir}")

    print("\n【5】微调之后模型评估")
    ft_metric = evaluate_embedding(out_model_dir, eval_qa_path, corpus_texts)

    print("\n==========微调前后指标对比==========")
    print(f"微调前：{base_metric}")
    print(f"微调后：{ft_metric}")
    print("验收条件：微调后 recall@1 / recall@3 高于微调前")

if __name__ == "__main__":
    main()
