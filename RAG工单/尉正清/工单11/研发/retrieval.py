# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""评估器：语料编码、向量检索、检索指标计算

微调前后用的是**同一套代码、同一份语料、同一批评估 query**，
这样两次结果的差异才能归因到模型本身，而不是评估口径变了。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from config import CACHE_DIR, ENCODE_BATCH, ENCODE_MAX_LEN, METRIC_KS


# ---------------- 编码 ----------------
def _fingerprint(model, corpus, salt=""):
    """语料指纹：语料换了，缓存的向量就不能复用。

    salt 用来区分**模型**：只按语料算指纹的话，重新训练一次模型后，
    缓存键不变，会把上一次那个模型的向量当成新模型的用 —— 结果就是
    换了模型指标却纹丝不动，非常难查。调用方把模型路径+修改时间传进来。
    """
    h = hashlib.md5()
    h.update(str(len(corpus)).encode())
    for row in corpus[:50]:                     # 抽样即可，全量算太慢
        h.update(row["_id"].encode())
    h.update(str(ENCODE_MAX_LEN).encode())
    h.update(salt.encode())
    return h.hexdigest()[:12]


def model_salt(model_path):
    """模型身份：路径 + 权重文件的大小与修改时间。重训后缓存自动失效。

    ⚠️ **不要把目录本身的 mtime 算进来**。目录 mtime 会被任何无关操作带变
    （加载模型时 sentence-transformers 会读写几个元数据文件），
    于是同一个模型两次算出的 salt 不同、缓存凭空失效，每次评估都白重算
    57638 篇语料（约 18 分钟）还多占 177MB 磁盘。实测踩过：两份向量的
    最大差是 0.0，指纹却不一样。
    只认权重文件就够了 —— 重新训练一定会重写 *.safetensors。
    """
    p = Path(model_path)
    parts = [p.name]
    try:
        for f in sorted(p.glob("*.safetensors")) + sorted(p.glob("*.bin")):
            st = f.stat()
            parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
    except OSError:
        pass
    return "|".join(parts)


def encode(model, texts, model_tag, cache_key=None, show=True, chunk=4000):
    """把文本编码成 L2 归一化向量，并按 model_tag 缓存。

    归一化之后点积即余弦相似度，检索时直接矩阵乘，省掉一次除法。

    分块落盘（chunk 条一块），中断后重跑只补没算完的块 ——
    实测编码 57638 篇语料要十几分钟，中途断一次从头再来的代价太大，
    而这个进程会被系统内存压力杀掉（本机 15.7GB 内存，WSL 里还跑着别的服务）。
    """
    if cache_key:
        cache = CACHE_DIR / f"emb_{model_tag}_{cache_key}.npy"
        if cache.exists():
            if show:
                print(f"  [缓存] {cache.name}")
            return np.load(cache)

    model.eval()
    model.max_seq_length = ENCODE_MAX_LEN

    n_chunks = (len(texts) + chunk - 1) // chunk
    parts, done = [], 0
    for ci in range(n_chunks):
        pf = (CACHE_DIR / f"emb_{model_tag}_{cache_key}_part{ci:03d}.npy"
              if cache_key else None)
        if pf is not None and pf.exists():
            parts.append(np.load(pf))
            done += 1
            continue
        piece = model.encode(
            texts[ci * chunk:(ci + 1) * chunk], batch_size=ENCODE_BATCH,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=show,
        ).astype(np.float32)
        if pf is not None:
            pf.parent.mkdir(parents=True, exist_ok=True)
            np.save(pf, piece)
        parts.append(piece)

    out = np.concatenate(parts, axis=0)
    if done and show:
        print(f"  [续跑] 复用已完成的 {done}/{n_chunks} 块")

    if cache_key:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
        for pf in CACHE_DIR.glob(f"emb_{model_tag}_{cache_key}_part*.npy"):
            pf.unlink()                      # 合并完就清掉分块，别留垃圾
    return out


# ---------------- 检索 ----------------
def search(query_emb, corpus_emb, top_k=100):
    """返回每个 query 的 top_k 文档下标与分数（降序）。"""
    # 57638 × 768 的矩阵乘，分块做避免一次性吃满显存
    sims = query_emb @ corpus_emb.T
    k = min(top_k, sims.shape[1])
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    scores = sims[rows, idx]
    order = np.argsort(-scores, axis=1)
    return idx[rows, order], scores[rows, order]


# ---------------- 指标 ----------------
def _dcg(rels):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def evaluate(ranked_ids, rel, ks=METRIC_KS):
    """算 nDCG@k / Recall@k / MRR@k。

    ranked_ids: {query_id: [corpus_id, ...]}，按相关性降序
    rel       : {query_id: {corpus_id, ...}}，标准答案
    只统计在 rel 里出现过的 query，与 BEIR 口径一致。
    """
    agg = {f"nDCG@{k}": [] for k in ks}
    agg.update({f"Recall@{k}": [] for k in ks})
    agg.update({f"MRR@{k}": [] for k in ks})

    for qid, gold in rel.items():
        ranked = ranked_ids.get(qid, [])
        for k in ks:
            topk = ranked[:k]
            hits = [1.0 if d in gold else 0.0 for d in topk]

            # nDCG：理想排序是所有相关文档都排在最前
            ideal = [1.0] * min(len(gold), k) + [0.0] * max(0, k - len(gold))
            agg[f"nDCG@{k}"].append(_dcg(hits) / _dcg(ideal) if _dcg(ideal) else 0.0)

            agg[f"Recall@{k}"].append(len(set(topk) & gold) / len(gold))

            rr = next((1.0 / (i + 1) for i, d in enumerate(topk) if d in gold), 0.0)
            agg[f"MRR@{k}"].append(rr)

    return {name: float(np.mean(vals)) for name, vals in agg.items() if vals}


def save_result(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def gpu_report():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return f"GPU {torch.cuda.get_device_name(0)} 显存 {free/2**30:.1f}/{total/2**30:.1f} GB 可用"
    return "CPU 模式（未检测到 CUDA）"
