# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
评估模块：对比微调前后 Embedding 模型的检索效果。
指标：Top-3 检索命中率（相关文本块是否被召回）。
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from data_gen import load_chunks


def retrieve_hit(model, query, corpus_emb, corpus, top_k=3, keyword=None):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = corpus_emb @ q
    order = np.argsort(-scores)[:top_k]
    for i in order:
        if keyword and keyword in corpus[i]:
            return True
    return False


def evaluate_model(model):
    chunks = load_chunks()
    corpus_emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    pairs = [
        ("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？", "军用"),
        ("武汉兴图新科电子股份有限公司法定代表人是谁？", "法定代表人"),
        ("武汉力源信息技术股份有限公司本次发行股数是多少？", "发行股数"),
        ("武汉兴图新科电子股份有限公司注册资本是多少？", "注册资本"),
    ]
    hits = sum(retrieve_hit(model, q, corpus_emb, chunks, top_k=3, keyword=k) for q, k in pairs)
    return {"hit@3": round(hits / len(pairs), 3)}


def main():
    print("==== 微调前评估 ====")
    base = SentenceTransformer(config.BASE_MODEL)
    print("基础模型:", evaluate_model(base))

    print("==== 微调后评估 ====")
    if os.path.exists(config.OUTPUT_MODEL_DIR):
        tuned = SentenceTransformer(config.OUTPUT_MODEL_DIR)
        print("微调模型:", evaluate_model(tuned))
    else:
        print("未找到微调模型，请先运行 train.py")


if __name__ == "__main__":
    main()
