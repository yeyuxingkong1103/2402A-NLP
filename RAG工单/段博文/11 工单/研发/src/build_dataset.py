# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
"""
数据集生成模块（三）：训练/评估集划分 + BM25 困难负例挖掘。

输入：dataset/qa_pairs.jsonl
输出：
    dataset/train.jsonl   —— {query, positive, hard_negative} 三元组
    dataset/eval.jsonl    —— {query, positive}
    dataset/corpus.jsonl  —— 评估语料（全部段落）
"""

import json
import os
import random
from collections import OrderedDict

import jieba
from rank_bm25 import BM25Okapi

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")

EVAL_SIZE = 60


def tokenize(text: str):
    return list(jieba.cut(text))


def mine_hard_negatives(qa_list, all_passages):
    """用 BM25 为每条 QA 挖 1 个困难负例（BM25 高分但非正例的段落）。"""
    # 建立 BM25 索引
    corpus_tokens = [tokenize(p) for p in all_passages]
    bm25 = BM25Okapi(corpus_tokens)

    train_with_neg = []
    for qa in qa_list:
        pos_passage = qa["passage"]
        scores = bm25.get_scores(tokenize(qa["question"]))

        # 取 BM25 排名前 20 中第一个非正例的段落作为 hard negative
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
        hard_neg = None
        for idx in top_idx:
            if all_passages[idx] != pos_passage:
                hard_neg = all_passages[idx]
                break
        if hard_neg is None:
            hard_neg = random.choice([p for p in all_passages if p != pos_passage])

        train_with_neg.append({
            "query": qa["question"],
            "positive": pos_passage,
            "hard_negative": hard_neg,
            "answer": qa["answer"],
        })

    return train_with_neg


def main():
    qa_list = []
    with open(os.path.join(DATASET, "qa_pairs.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            qa_list.append(json.loads(line))

    print(f"共加载 {len(qa_list)} 条 QA")

    # 打乱后划分
    random.shuffle(qa_list)
    eval_qa = qa_list[:EVAL_SIZE]
    train_qa = qa_list[EVAL_SIZE:]

    # 全部段落（评估语料），去重保序
    all_passages = list(OrderedDict.fromkeys(
        qa["passage"] for qa in qa_list
    ))
    print(f"语料段落数（去重）: {len(all_passages)}")

    # 训练集挖困难负例（训练三元组）
    print("挖掘 BM25 困难负例...")
    train_data = mine_hard_negatives(train_qa, all_passages)

    # 评估集
    eval_data = [{"query": qa["question"], "positive": qa["passage"], "answer": qa["answer"]}
                 for qa in eval_qa]

    # 写出
    with open(os.path.join(DATASET, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(DATASET, "eval.jsonl"), "w", encoding="utf-8") as f:
        for r in eval_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(DATASET, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")

    print(f"\n数据集划分完成：")
    print(f"  train: {len(train_data)} 条（三元组：query/positive/hard_negative）")
    print(f"  eval:  {len(eval_data)} 条")
    print(f"  corpus: {len(all_passages)} 个段落")


if __name__ == "__main__":
    main()
