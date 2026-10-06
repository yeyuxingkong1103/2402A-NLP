# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
模块：微调前后指标可视化
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_compare(before_path="eval_before.json", after_path="eval_after.json",
                  output="eval_compare.png"):
    with open(before_path, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(after_path, "r", encoding="utf-8") as f:
        after = json.load(f)

    def normalize_key(k):
        k = k.replace("eval_", "").replace("finetune_eval_", "").replace("cosine_", "")
        return k

    before_norm = {normalize_key(k): v for k, v in before.items()}
    after_norm = {normalize_key(k): v for k, v in after.items()}

    key_metrics = [
        ("accuracy@1", "Accuracy@1"),
        ("accuracy@5", "Accuracy@5"),
        ("accuracy@10", "Accuracy@10"),
        ("recall@10", "Recall@10"),
        ("ndcg@10", "NDCG@10"),
        ("mrr@10", "MRR@10"),
        ("map@100", "MAP@100"),
    ]

    labels = []
    before_vals = []
    after_vals = []
    for k, label in key_metrics:
        if k in before_norm or k in after_norm:
            labels.append(label)
            before_vals.append(before_norm.get(k, 0))
            after_vals.append(after_norm.get(k, 0))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, before_vals, width, label="Before", color="#4A90E2")
    bars2 = ax.bar(x + width/2, after_vals, width, label="After", color="#E94B3C")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Embedding Fine-tune: Before vs After")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    print(f"✅ 已保存到 {output}")


if __name__ == "__main__":
    if os.path.exists("eval_before.json") and os.path.exists("eval_after.json"):
        plot_compare()
    else:
        print("⚠️ 请先运行 finetune.py")
