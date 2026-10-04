# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
微调主程序：数据集与模型加载 → 定义损失函数 → 定义训练参数 →
创建评估器 → 微调前评估 → 训练 → 保存模型（微调后评估见 evaluate.py）。
"""
import json
import os

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

import config
from data_gen import generate_dataset, load_chunks


def load_dataset():
    path = os.path.join(config.DATA_DIR, "train_pairs.jsonl")
    if not os.path.exists(path):
        generate_dataset()
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def build_examples(samples, loss_type):
    examples = []
    for s in samples:
        if loss_type == "triplet":
            examples.append(InputExample(texts=[s["query"], s["positive"], s["negative"]]))
        elif loss_type in ("contrastive", "cosine"):
            examples.append(InputExample(texts=[s["query"], s["positive"]], label=1.0))
            examples.append(InputExample(texts=[s["query"], s["negative"]], label=0.0))
        elif loss_type == "matryoshka":
            examples.append(InputExample(texts=[s["query"], s["positive"], s["negative"]]))
    return examples


def get_loss(model, loss_type):
    if loss_type == "triplet":
        return losses.TripletLoss(model)               # 三元组损失
    if loss_type == "contrastive":
        return losses.ContrastiveLoss(model)            # 对比损失
    if loss_type == "cosine":
        return losses.CosineSimilarityLoss(model)       # 余弦相似度损失
    if loss_type == "matryoshka":
        return losses.MatryoshkaLoss(model, [768, 512, 256, 128, 64])  # 套娃损失
    raise ValueError("unknown loss_type: " + loss_type)


def build_evaluator(model, loss_type):
    chunks = load_chunks()
    queries = {s["query"]: s["positive"] for s in load_dataset()}
    return evaluation.InformationRetrievalEvaluator(
        queries={q: p for q, p in list(queries.items())[:20]},
        corpus={str(i): c for i, c in enumerate(chunks[:200])},
        corpus_chunk_size=500,
    )


def main():
    samples = load_dataset()
    examples = build_examples(samples, config.LOSS_TYPE)
    loader = DataLoader(examples, shuffle=True, batch_size=config.BATCH_SIZE)

    # 数据集与模型加载
    model = SentenceTransformer(config.BASE_MODEL)

    # 定义损失函数
    loss = get_loss(model, config.LOSS_TYPE)

    # 创建评估器
    evaluator = build_evaluator(model, config.LOSS_TYPE)

    # 微调前评估
    print("[train] 微调前评估...")
    before = evaluator(model)
    print("[train] 微调前评估结果:", before)

    # 训练（定义训练参数）
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=config.EPOCHS,
        warmup_steps=config.WARMUP_STEPS,
        optimizer_params={"lr": config.LEARNING_RATE},
        evaluator=evaluator,
        evaluation_steps=config.EVAL_STEPS,
        output_path=config.OUTPUT_MODEL_DIR,
    )

    # 微调后评估
    print("[train] 微调后评估...")
    after = evaluator(model)
    print("[train] 微调后评估结果:", after)
    print("[train] 模型已保存到", config.OUTPUT_MODEL_DIR)


if __name__ == "__main__":
    main()
