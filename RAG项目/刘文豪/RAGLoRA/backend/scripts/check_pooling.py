# -*- coding: utf-8 -*-
"""核对 bge-m3 的池化方式：CLS vs mean。

⚠️ 为什么值得单独测
===================
`bge-m3/1_Pooling/config.json` 声明的是 **CLS 池化**：

    {"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}

但本项目 `app/services/embed.py` 实现的是 **mean pooling**：

    dense = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

也就是说**生产链路一直在用与模型设计不符的池化方式**。
系统跑出了 100% 来源命中率，说明 mean pooling 不算错到家，
但这是一个从未被验证过的偏离 —— 而且它直接决定微调该按哪种池化训练
（若微调用 CLS 训练、上线用 mean 推理，二者不匹配，微调收益会被抵消）。

本脚本用同一批语料与问题实测两种池化的检索命中率，给出依据。

用法
----
    python scripts/check_pooling.py                    # 默认 800 条语料
    python scripts/check_pooling.py --corpus 2000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config                      # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
MODEL_DIR = Path(config.EMBED_MODEL_PATH)


def load_corpus(limit: int):
    from app.services import milvus_store
    c = milvus_store.get_client()
    rows, off = [], 0
    while True:
        r = c.query(collection_name="kb_legal", filter="",
                    output_fields=["text", "law_name", "source"],
                    limit=1000, offset=off)
        if not r:
            break
        rows.extend(r)
        off += len(r)
        if limit and len(rows) >= limit:
            break
        if len(r) < 1000:
            break
    rows = rows[:limit] if limit else rows
    return ([(x.get("text") or "") for x in rows],
            [(x.get("law_name") or x.get("source") or "") for x in rows])


def encode(texts: list[str], pooling: str, batch: int = 16,
           max_len: int = 256) -> np.ndarray:
    """按指定池化方式编码。pooling: 'cls' | 'mean'"""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model = model.eval().to("cpu")

    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            if pooling == "cls":
                vec = hidden[:, 0]                      # 取 [CLS]
            else:
                mask = enc["attention_mask"].unsqueeze(-1)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.float().numpy())

    del model
    return np.vstack(out)


def hit_rate(vecs, sources, q_vecs, expects, topk=5):
    sims = q_vecs @ vecs.T
    hits = 0
    for i, exp in enumerate(expects):
        idx = np.argsort(-sims[i])[:topk]
        got = " ".join(sources[j] for j in idx)
        if any(e.lower() in got.lower() for e in exp):
            hits += 1
    return hits, len(expects)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=800)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    qa = json.loads((EVAL_DIR / "qa_set.json").read_text(encoding="utf-8"))
    questions = [x["q"] for x in qa["legal"]]
    expects = [x.get("expect_sources", []) for x in qa["legal"]]

    print("=" * 70)
    print(f"池化方式对比 | 语料 {args.corpus} 条 | 问题 {len(questions)} 条 | top-{args.topk}")
    print("=" * 70)

    texts, sources = load_corpus(args.corpus)
    print(f"载入语料 {len(texts)} 条\n")

    results = {}
    for pooling in ("cls", "mean"):
        t = time.time()
        vecs = encode(texts, pooling)
        q_vecs = encode(questions, pooling)
        h, n = hit_rate(vecs, sources, q_vecs, expects, args.topk)
        results[pooling] = h / n * 100
        print(f"  {pooling:<5} 池化 -> 来源命中率 {h}/{n} = {h/n*100:.1f}%  "
              f"（{time.time()-t:.0f}s）")

    print()
    print("-" * 70)
    best = max(results, key=results.get)
    print(f"  结论：{'CLS' if best=='cls' else 'mean'} 池化更优（{results[best]:.1f}%）")
    if abs(results['cls'] - results['mean']) < 2:
        print("  ⚠️ 两者差距在噪声范围内（<2pp），不足以支撑改变生产配置")
    print()
    print("  生产链路（embed.py）当前用的是 **mean** 池化。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
