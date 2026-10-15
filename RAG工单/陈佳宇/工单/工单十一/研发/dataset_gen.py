# dataset_gen.py 工单11 生成embedding微调三元组数据集
# 工单编号：人工智能NLP‑RAG 项目‑Embedding 模型微调任务
import json
import os

# 复用金融招股模拟文档片段（和前面工单业务数据保持一致）
finance_texts = [
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

qa_pairs = [
    {"q":"公司2024年营业收入多少？","pos":"2024年公司营业收入26.8亿元，同比增长12.3%。"},
    {"q":"研发部有多少员工？","pos":"研发部主要负责芯片方案设计，现有员工126人。"},
    {"q":"销售部有哪些下属部门？","pos":"销售部包含两大业务部门：大客户销售部、中小客户销售部。"},
    {"q":"公司竞争对手是谁？","pos":"主要竞争对手：中电港、文晔科技。"},
    {"q":"募投项目有哪些？","pos":"公司募投项目包括：半导体测试中心建设项目、研发中心升级项目。"}
]

def build_triple_dataset(text_list, save_path):
    triples = []
    n = len(text_list)
    for i in range(n):
        anchor = text_list[i]
        # 正例：取相邻同领域片段
        pos_idx = (i+1) % n
        positive = text_list[pos_idx]
        # 负例：尽量隔远，语义无关片段
        neg_idx = (i + 6) % n
        negative = text_list[neg_idx]
        triples.append({
            "anchor": anchor,
            "positive": positive,
            "negative": negative
        })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path,"w",encoding="utf-8") as f:
        for item in triples:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    print(f"生成三元组数据集完成，共{len(triples)}条，保存至 {save_path}")
    return triples

def build_eval_qa(save_path):
    """构建评估用QA集合"""
    eval_data = []
    for item in qa_pairs:
        eval_data.append(item)
    with open(save_path,"w",encoding="utf-8") as f:
        json.dump(eval_data,f,ensure_ascii=False,indent=2)
    print(f"评估QA数据集保存至 {save_path}")

if __name__ == "__main__":
    build_triple_dataset(finance_texts, save_path="./data/train_triple.jsonl")
    build_eval_qa(save_path="./data/eval_qa.json")
