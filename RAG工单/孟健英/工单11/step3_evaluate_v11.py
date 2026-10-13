# 工单编号：人工智能 NLP-RAG-Embedding 模型微调任务
import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
from config_v11 import (
    CHUNK_FILE, DATASET_FILE, MODEL_OUTPUT_DIR, BASE_MODEL, EVAL_RESULT_FILE
)

DEVICE = "cuda"   # 改成 "cuda" 会快 5~10 倍
MAX_CORPUS = 800  # 限制 corpus 规模，避免 CPU 编码 7000+ 条太慢


def evaluate(model, queries, positives, corpus_texts, corpus_embs, k_list=(1, 5, 10)):
    results = {f"hit@{k}": 0 for k in k_list}
    mrr_sum = 0.0
    text_to_idx = {t: i for i, t in enumerate(corpus_texts)}

    for q, pos_text in zip(queries, positives):
        q_emb = model.encode([q], normalize_embeddings=True)
        sims = util.cos_sim(q_emb, corpus_embs)[0]
        ranked = np.argsort(sims.numpy())[::-1]

        pos_idx = text_to_idx.get(pos_text)
        if pos_idx is None:
            continue
        rank = int(np.where(ranked == pos_idx)[0][0]) + 1

        for k in k_list:
            if rank <= k:
                results[f"hit@{k}"] += 1
        mrr_sum += 1.0 / rank

    n = len(queries)
    for k in k_list:
        results[f"hit@{k}"] /= n
    results["mrr@10"] = mrr_sum / n
    return results


if __name__ == "__main__":
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        triplets = json.load(f)
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)["chunks"]

    # 只用包含正例的 chunk 作为 corpus，并限制总量
    pos_texts = set(t["positive"] for t in triplets)
    corpus_texts = [c["text"] for c in all_chunks if c["text"] in pos_texts]
    if len(corpus_texts) < MAX_CORPUS:
        # 补充一些随机 chunk 让评估更真实
        extra = [c["text"] for c in all_chunks if c["text"] not in pos_texts]
        random.seed(42)
        random.shuffle(extra)
        corpus_texts += extra[:MAX_CORPUS - len(corpus_texts)]
    print(f"评估用 corpus：{len(corpus_texts)} 条")

    queries = [t["anchor"] for t in triplets]
    positives = [t["positive"] for t in triplets]
    print(f"评估用 query：{len(queries)} 条")

    # 微调前
    print("评估微调前模型...")
    base_model = SentenceTransformer(BASE_MODEL, device=DEVICE)
    corpus_embs_base = base_model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True)
    base_result = evaluate(base_model, queries, positives, corpus_texts, corpus_embs_base)
    print(f"微调前：{base_result}")

    # 微调后
    print("评估微调后模型...")
    ft_model = SentenceTransformer(MODEL_OUTPUT_DIR, device=DEVICE)
    corpus_embs_ft = ft_model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True)
    ft_result = evaluate(ft_model, queries, positives, corpus_texts, corpus_embs_ft)
    print(f"微调后：{ft_result}")

    with open(EVAL_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "before": base_result,
            "after": ft_result,
            "queries": len(queries),
            "corpus": len(corpus_texts)
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {EVAL_RESULT_FILE}")