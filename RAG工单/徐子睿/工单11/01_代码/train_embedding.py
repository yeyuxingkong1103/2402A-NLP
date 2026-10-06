# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-Embeddings 模型微调任务
# 模块：embedding_ft/train_embedding —— 在领域问答对上微调 Embedding 模型
# 说明：基座 BAAI/bge-base-zh-v1.5（本地缓存），损失 = MultipleNegativesRankingLoss
#       （in-batch negatives，等价 InfoNCE）；训练数据 = `embedding_ft/data/pairs.jsonl`。
#       保存到 `embedding_ft/model_bge_ft`（SentenceTransformer 格式）。
# 用法：python embedding_ft/train_embedding.py [--epochs 2] [--batch 16] [--lr 2e-5] [--limit 0]
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PAIRS = os.path.join(HERE, "data", "pairs.jsonl")
OUTDIR = os.path.join(HERE, "model_bge_ft")
BASE = os.environ.get("FT_BASE_MODEL", "BAAI/bge-base-zh-v1.5")


def load_pairs(limit=0):
    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append((d["query"], d["positive"][:600]))
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maxseq", type=int, default=256)
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import torch

    torch.set_num_threads(int(os.environ.get("FT_THREADS", "8")))
    print("torch threads =", torch.get_num_threads(), "| maxseq =", a.maxseq)

    rows = load_pairs(a.limit)
    rows = [(q, p[:400]) for q, p in rows]      # 正例截断，避免长尾拖慢 CPU 训练
    print("训练对 =", len(rows), "| 基座 =", BASE, "| epochs =", a.epochs, "| batch =", a.batch)
    model = SentenceTransformer(BASE)
    model.max_seq_length = a.maxseq             # 实际生效的最大序列长度（默认 512 → 256 提速）
    print("基座加载完成，向量维度 =", model.get_sentence_embedding_dimension(),
          "| 可训练参数 =", sum(p.numel() for p in model.parameters()))

    examples = [InputExample(texts=[q, p]) for q, p in rows]
    loader = DataLoader(examples, shuffle=True, batch_size=a.batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(loader) * a.epochs * 0.1)

    t0 = time.time()
    model.fit(train_objectives=[(loader, loss)], epochs=a.epochs,
              warmup_steps=max(1, warmup), optimizer_params={"lr": a.lr},
              output_path=OUTDIR, show_progress_bar=True,
              use_amp=False)
    print("训练完成，用时 %.1f 分钟" % ((time.time() - t0) / 60))
    model.save(OUTDIR)
    print("saved ->", OUTDIR)
    print("EMBED_FT_DONE")


if __name__ == "__main__":
    main()
