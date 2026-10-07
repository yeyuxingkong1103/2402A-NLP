# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化 - 可视化"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BEFORE = {
    "vector": 0.0353, "bm25": 0.0286, "reranker": 0.0345, "graph": 0.0066,
    "total": 0.1211,
}
AFTER = {
    "vector": 0.0344, "bm25": 0.0263, "reranker": 0.0239, "graph": 0.0066,
    "total": 0.0846,
}


def plot_comparison():
    stages = ["vector", "bm25", "reranker", "graph", "total"]
    labels = ["Vector (bge-small)", "BM25 (jieba)", "Reranker (bge-base)",
              "Graph RAG", "Total"]
    before_vals = [BEFORE[s] for s in stages]
    after_vals = [AFTER[s] for s in stages]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, before_vals, width, label="Before", color="#4A90E2")
    bars2 = ax.bar(x + width/2, after_vals, width, label="After", color="#E94B3C")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("RAG Performance: Before vs After Optimization", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.4f}",
                ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.4f}",
                ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("perf_compare.png", dpi=120)
    print("OK: perf_compare.png")


def plot_waterfall():
    stages = ["Query", "Vector", "BM25", "Reranker", "Graph", "Post"]
    times = [0.0004, 0.0353, 0.0286, 0.0345, 0.0066, 0.0003]
    colors = ["#50C878", "#4A90E2", "#9B59B6", "#E94B3C", "#F5A623", "#95A5A6"]

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = 0
    for i, (stage, t, color) in enumerate(zip(stages, times, colors)):
        ax.bar(stage, t, bottom=bottom, color=color, alpha=0.8)
        ax.text(i, bottom + t/2, f"{t:.4f}s\n({t/sum(times)*100:.1f}%)",
                ha="center", va="center", fontsize=9, color="white", weight="bold")
        bottom += t

    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(f"RAG Pipeline Breakdown (Total: {sum(times):.4f}s)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("perf_waterfall.png", dpi=120)
    print("OK: perf_waterfall.png")


def plot_pie():
    labels = ["Vector", "BM25", "Reranker", "Graph", "Other"]
    sizes = [0.0353, 0.0286, 0.0345, 0.0066, 0.0007]
    colors = ["#4A90E2", "#9B59B6", "#E94B3C", "#F5A623", "#95A5A6"]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("RAG Pipeline Bottleneck Distribution", fontsize=14)

    plt.tight_layout()
    plt.savefig("perf_pie.png", dpi=120)
    print("OK: perf_pie.png")


if __name__ == "__main__":
    plot_comparison()
    plot_waterfall()
    plot_pie()
    print()
    print("All charts generated.")
