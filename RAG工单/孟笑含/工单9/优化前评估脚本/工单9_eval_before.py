# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Graph RAG 优化任务
模块：RAGAS 评估（优化前基线）
"""

import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from rag_qa_system import RAGQASystem
from local_llm import LocalQwenLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

QUESTIONS = [
    {"id": 1, "question": "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？", "ground_truth": "平安银行2019年盈利增长主要得益于：业务结构优化、零售业务收入贡献支柱地位、对公和资金同业业务步入正轨、营收净利润增速创新高、逾期60天以上贷款占比和拨备覆盖率不断优化。"},
    {"id": 2, "question": "招商银行在其2019年年报中提到的创新商业模式有哪些？", "ground_truth": "招商银行2019年探索的创新商业模式包括：聚焦高频生活场景，与合作伙伴共建生态系统；发起设立逾越者联盟，支持开放的新生态建设；跨界参与咖啡零售、出行预订和影票销售等场景。"},
    {"id": 3, "question": "平安银行的风险管理体系如何体现其应对宏观经济周期波动的能力？", "ground_truth": "平安银行通过动态调整拨备覆盖率、加强资产质量控制、优化贷款结构（减少高风险行业、增加抗周期行业如基础设施、绿色金融）等措施应对经济周期波动。"},
    {"id": 4, "question": "分析这些银行和保险公司在面对经济周期波动时的共同策略与差异化策略。", "ground_truth": "共同策略：动态拨备覆盖率调整、信用风险监控、资本结构优化、新兴领域投资（绿色金融、科技金融）。差异化：银行关注逾期贷款管理，保险公司侧重长期资产配置和负债久期管理。"},
    {"id": 5, "question": "平安银行2019年营业收入是多少？", "ground_truth": "1,379亿元"},
    {"id": 6, "question": "中国平安2019年归属于母公司股东的净利润是多少？", "ground_truth": "1,494.07亿元"},
    {"id": 7, "question": "招商银行2019年不良贷款率是多少？", "ground_truth": "1.16%"},
    {"id": 8, "question": "中信证券2020年营业收入是多少？", "ground_truth": "543.83亿元"},
    {"id": 9, "question": "中国人寿2020年保费收入是多少？", "ground_truth": "6,120.65亿元"},
    {"id": 10, "question": "国泰君安2021年营业收入是多少？", "ground_truth": "428.17亿元"},
]

PDF_PATHS = [
    "./data/ccf_competition/pdf/2019年_平安银行_000001_年度报告.pdf",
    "./data/ccf_competition/pdf/2019年_中国平安_601318_年度报告.pdf",
    "./data/ccf_competition/pdf/2019年_招商银行_600036_年度报告.pdf",
    "./data/ccf_competition/pdf/2019年_邮储银行_601658_年度报告.pdf",
    "./data/ccf_competition/pdf/2020年_中信证券_600030_年度报告.pdf",
    "./data/ccf_competition/pdf/2020年_中国人寿_601628_年度报告.pdf",
    "./data/ccf_competition/pdf/2021年_招商证券_600999_年度报告.pdf",
    "./data/ccf_competition/pdf/2021年_中国太平_601601_年度报告.pdf",
    "./data/ccf_competition/pdf/2021年_国泰君安_601211_年度报告.pdf",
]


if __name__ == "__main__":
    print("=" * 70)
    print("【工单9】RAGAS 评估 - 优化前基线")
    print("=" * 70)

    system = RAGQASystem(PDF_PATHS)

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q in QUESTIONS:
        r = system.answer(q["question"], use_rag=True)
        contexts = [c["content"] for c in r.get("retrieved_contexts", [])]
        data["question"].append(q["question"])
        data["answer"].append(r["answer"])
        data["contexts"].append(contexts)
        data["ground_truth"].append(q["ground_truth"])
        print(f"  ID:{q['id']} 完成 {r['response_time']}s")

    dataset = Dataset.from_dict(data)
    llm = LocalQwenLLM()

    # 加载本地 BGE embedding（RAGAS 默认用 OpenAI，需手动传）
    print("正在加载本地 embedding：BAAI/bge-m3...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={
            "device": "cuda",
            "use_safetensors": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )

    print("\n正在 RAGAS 评估（约 5~10 分钟）...")
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    print("\n" + "=" * 70)
    print("RAGAS 评估结果（优化前）")
    print("=" * 70)
    print(result)

    result_dict = {
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
    }
    with open("工单9_评估_优化前.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print("\n✅ 已保存到 工单9_评估_优化前.json")
