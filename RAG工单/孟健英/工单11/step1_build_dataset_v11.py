# 工单编号：人工智能 NLP-RAG-Embedding 模型微调任务
import json
import random
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from config_v11 import (
    CHUNK_FILE, DATASET_FILE, BASE_MODEL, MAX_QUERIES, NEG_PER_QUERY,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
)

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def gen_query(chunk_text):
    """用 DeepSeek 为一个 chunk 生成一个自然语言 query"""
    prompt = f"""根据下面这段金融年报内容，生成一个用户可能会问的问题。
要求：问题要具体，能通过这段内容回答。
只输出问题，不要解释。

内容：{chunk_text[:800]}
问题："""
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()


def mine_negatives(query_emb, all_embs, pos_idx, k=NEG_PER_QUERY):
    """挖掘与 query 最不相似的 chunk 作为负例"""
    sims = util.cos_sim(query_emb, all_embs)[0]
    sims[pos_idx] = -1  # 排除正例
    neg_idx = np.argsort(sims.numpy())[:k]
    return neg_idx.tolist()


if __name__ == "__main__":
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    random.shuffle(chunks)
    chunks = chunks[:MAX_QUERIES]
    print(f"使用 {len(chunks)} 个 chunk 生成 query...")

    # 先编码所有 chunk，用于负例挖掘
    model = SentenceTransformer(BASE_MODEL, device="cpu")
    texts = [c["text"] for c in chunks]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    triplets = []
    for i, chunk in enumerate(chunks):
        try:
            q = gen_query(chunk["text"])
        except Exception as e:
            print(f"跳过 {i}: {e}")
            continue

        q_emb = model.encode([q], normalize_embeddings=True)
        neg_idx = mine_negatives(q_emb, embs, i)

        for n in neg_idx:
            triplets.append({
                "anchor": q,
                "positive": chunk["text"],
                "negative": chunks[n]["text"]
            })

        if (i + 1) % 50 == 0:
            print(f"已生成 {i+1}/{len(chunks)} 条 query，累计 {len(triplets)} 个三元组")

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(triplets)} 个三元组到 {DATASET_FILE}")