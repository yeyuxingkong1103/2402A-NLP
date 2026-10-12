# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
数据集生成模块：
  1. 解析招股说明书 PDF，切分文本块；
  2. 生成“问答对”训练数据（正例对、负例对、三元组）；
  3. 保存为 JSONL 数据集，供 train.py 加载。
"""
import json
import os
import random

import pymupdf

import config


def chunk_text(text, size=500, overlap=80):
    text = text.replace("\x00", " ").strip()
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c]


def load_chunks():
    chunks = []
    for pdf in [config.PDF1, config.PDF2]:
        if not os.path.exists(pdf):
            continue
        doc = pymupdf.open(pdf)
        text = "\n".join(p.get_text() for p in doc)
        doc.close()
        chunks.extend(chunk_text(text))
    return chunks


def generate_dataset():
    """生成训练数据集：每个样本含 query/positive/negative。"""
    chunks = load_chunks()
    random.seed(42)

    # 一组固定的金融问答 query（模拟真实用户问题）
    queries = [
        "公司的法定代表人是谁？",
        "公司的注册资本是多少？",
        "报告期内军用领域的收入是多少？",
        "本次发行股数是多少？",
        "募集资金拟投资哪些项目？",
        "存在控制关系的关联方有哪些？",
    ]

    samples = []
    for q in queries:
        # 正例：包含关键词的文本块
        pos_candidates = [c for c in chunks if any(k in c for k in ["法定代表人", "注册资本", "军用", "发行股数", "募集资金", "关联方"])]
        if not pos_candidates:
            continue
        positive = random.choice(pos_candidates)
        negatives = [c for c in chunks if c != positive]
        negative = random.choice(negatives)
        samples.append({"query": q, "positive": positive, "negative": negative})

    os.makedirs(config.DATA_DIR, exist_ok=True)
    out = os.path.join(config.DATA_DIR, "train_pairs.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[data_gen] 生成 {len(samples)} 条训练样本 -> {out}")
    return samples


if __name__ == "__main__":
    generate_dataset()
