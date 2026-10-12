# -*- coding: utf-8 -*-
"""
finetune.py — Embedding 模型微调
工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务

在金融招股说明书数据上微调嵌入模型（默认 BAAI/bge-small-zh-v1.5），
覆盖工单要求的完整流程：
  1. 数据集生成 / 问答对生成
  2. 数据集与模型加载
  3. 定义损失函数（TripletLoss / MultipleNegativesRankingLoss / MatryoshkaLoss）
  4. 定义训练参数
  5. 创建评估器
  6. 微调前评估模型
  7. 微调后评估模型

运行（需安装 sentence-transformers + torch）：
  python finetune.py --epochs 3 --batch-size 8
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import Config  # noqa: E402
from dataset import build_dataset  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
PDF_PATHS = [
    os.path.join(ATTACH_DIR, "招股说明书1-无水印.pdf"),
    os.path.join(ATTACH_DIR, "招股说明书2.pdf"),
]
PDF_PATHS = [p for p in PDF_PATHS if os.path.exists(p)]

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
OUTPUT_DIR = "output/bge-small-zh-finance"


def evaluate_model(model, samples, prefix: str):
    """用 InformationRetrievalEvaluator 评估检索效果（NDCG/MAP/MRR/Recall）。"""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    # 构造：query -> {doc_id: text}，以及正例映射
    queries, corpus, relevant_docs = {}, {}, {}
    for i, s in enumerate(samples[:200]):
        qid = f"q{i}"
        queries[qid] = s["anchor"]
        corpus[f"d{i}"] = s["positive"]
        relevant_docs[qid] = {f"d{i}"}
    evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name=f"finance-{prefix}",
    )
    scores = evaluator(model, output_path=None)
    print(f"[{prefix}] NDCG@10={scores.get('finance-' + prefix + '_cosine_ndcg@10', 0):.4f} "
          f"Recall@10={scores.get('finance-' + prefix + '_cosine_recall@10', 0):.4f}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-llm", action="store_true", help="用 LLM 生成更自然的问答对")
    args = parser.parse_args()

    config = Config()

    # 1. 数据集生成
    print("[1/7] 数据集生成……")
    samples = build_dataset(PDF_PATHS, config, use_llm=args.use_llm, per_doc=300)
    print(f"      生成 {len(samples)} 条样本")

    # 2. 数据集与模型加载
    print(f"[2/7] 加载模型 {MODEL_NAME} ……")
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    model = SentenceTransformer(MODEL_NAME)

    train_examples = [
        InputExample(texts=[s["anchor"], s["positive"], s["negative"]]) for s in samples
    ]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # 3. 定义损失函数（三元组损失；可替换为 MNR / Matryoshka）
    print("[3/7] 定义损失函数：TripletLoss")
    train_loss = losses.TripletLoss(model=model)

    # 4. 定义训练参数
    print(f"[4/7] 训练参数：epochs={args.epochs}, batch={args.batch_size}")
    warmup = max(int(len(train_loader) * args.epochs * 0.1), 1)
    train_args = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": args.epochs,
        "warmup_steps": warmup,
        "evaluation_steps": 500,
        "save_steps": 500,
        "report_to": None,
    }

    # 5. 创建评估器
    print("[5/7] 创建评估器（InformationRetrievalEvaluator）")

    # 6. 微调前评估
    print("[6/7] 微调前评估……")
    before = evaluate_model(model, samples, "before")

    # 7. 微调
    print("[7/7] 开始微调……")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        **train_args,
    )
    model.save(OUTPUT_DIR)

    # 微调后评估
    print("微调后评估……")
    after = evaluate_model(model, samples, "after")
    print(f"\n模型已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
