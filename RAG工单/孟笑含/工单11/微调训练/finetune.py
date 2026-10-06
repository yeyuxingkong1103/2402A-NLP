# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
模块：Embedding 模型微调
功能：用 MNRL 损失微调 bge-base-zh-v1.5
"""

import os
import json
import random
import torch
from typing import List, Dict
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    evaluation,
)
from sentence_transformers.training_args import BatchSamplers

os.environ['HF_HUB_DISABLE_XET'] = '1'

BASE_MODEL = "BAAI/bge-base-zh-v1.5"
TRAIN_DATA = "train_data.json"
OUTPUT_DIR = "finetuned_model"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataset(data):
    random.shuffle(data)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    eval_data = data[split:]
    train_dataset = Dataset.from_dict({
        "anchor": [d["query"] for d in train_data],
        "positive": [d["positive"] for d in train_data],
    })
    eval_dataset = Dataset.from_dict({
        "anchor": [d["query"] for d in eval_data],
        "positive": [d["positive"] for d in eval_data],
    })
    return train_dataset, eval_dataset, eval_data


def create_evaluator(eval_data):
    queries = {}
    corpus = {}
    relevant_docs = {}
    for i, d in enumerate(eval_data):
        qid = f"q{i}"
        did = f"d{i}"
        queries[qid] = d["query"]
        corpus[did] = d["positive"]
        relevant_docs[qid] = {did}
    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="finetune_eval",
        show_progress_bar=False,
    )


def main():
    print("=" * 60)
    print("Embedding 模型微调")
    print("=" * 60)

    print(f"\n[1/8] 加载数据集：{TRAIN_DATA}")
    data = load_data(TRAIN_DATA)
    print(f"  共 {len(data)} 对")

    print(f"\n[2/8] 加载模型：{BASE_MODEL}")
    try:
        model = SentenceTransformer(BASE_MODEL, model_kwargs={"use_safetensors": True})
    except Exception as e:
        print(f"⚠️ safetensors 加载失败，回退：{e}")
        model = SentenceTransformer(BASE_MODEL)
    print(f"  维度：{model.get_sentence_embedding_dimension()}")

    print("\n[3/8] 准备训练/评估数据集")
    train_dataset, eval_dataset, eval_data = prepare_dataset(data)
    print(f"  训练集：{len(train_dataset)}")
    print(f"  评估集：{len(eval_dataset)}")

    print("\n[4/8] 定义损失函数：MNRL")
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("\n[5/8] 创建评估器")
    evaluator = create_evaluator(eval_data)

    print("\n[6/8] 微调前评估...")
    baseline_metrics = evaluator(model)
    print("  微调前：")
    for k, v in baseline_metrics.items():
        print(f"    {k}: {v:.4f}")
    with open("eval_before.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, ensure_ascii=False, indent=2)

    print("\n[7/8] 开始微调...")
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_finetune_eval_cosine_ndcg@10",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        loss=train_loss, evaluator=evaluator,
    )
    trainer.train()

    print("\n[8/8] 微调后评估...")
    model.save(OUTPUT_DIR)
    print(f"  模型已保存：{OUTPUT_DIR}")
    after_metrics = evaluator(model)
    print("  微调后：")
    for k, v in after_metrics.items():
        print(f"    {k}: {v:.4f}")
    with open("eval_after.json", "w", encoding="utf-8") as f:
        json.dump(after_metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("对比")
    print("=" * 60)
    for k in baseline_metrics:
        b = baseline_metrics.get(k, 0)
        a = after_metrics.get(k, 0)
        print(f"{k:<40} 前={b:.4f} 后={a:.4f} 提升={a-b:+.4f}")


if __name__ == "__main__":
    main()
