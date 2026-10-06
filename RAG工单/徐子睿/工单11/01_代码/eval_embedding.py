# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-Embeddings 模型微调任务
# 模块：embedding_ft/eval_embedding —— 微调前后检索效果对比（同一语料、同一题集）
# 说明：对每个模型（微调前 base / 微调后 ft）重新编码全部语料块，再编码 16 道题，
#       按余弦相似度取 top-k，比对金标准页 → Hit@1 / Hit@3 / Hit@5 / MRR。
# 用法：python embedding_ft/eval_embedding.py
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

CHUNKS = os.path.join(ROOT, "data", "index", "chunks.jsonl")
HOLDOUT = os.path.join(HERE, "data", "eval_pairs.jsonl")
FTDIR = os.path.join(HERE, "model_bge_ft")
OUT_MD = os.path.join(HERE, "评估_微调前后.md")
OUT_JSON = os.path.join(HERE, "评估_微调前后.json")
BASE = os.environ.get("FT_BASE_MODEL", "BAAI/bge-base-zh-v1.5")


def load_corpus():
    docs, pages, ids = [], [], []
    with open(CHUNKS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("type") == "image":
                continue
            docs.append((c.get("text") or "")[:600])
            pages.append(str(c.get("page")))
            ids.append(c.get("id"))
    return docs, pages, ids


def load_questions():
    qs = []
    for p, key in (("evaluation/eval_questions.json", "page"),
                   ("evaluation/eval_questions_pdf2.json", "gold_page")):
        d = json.load(open(os.path.join(ROOT, p), encoding="utf-8"))
        for it in (d.get("questions") or d.get("items")):
            gp = set(str(x) for x in (it.get(key) or []))
            qs.append({"id": it["id"], "q": it["question"], "gold": gp, "src": os.path.basename(p)})
    return qs


def load_holdout():
    """留出页 + 新问法的泛化测试集（见 make_eval_set.py）。"""
    if not os.path.exists(HOLDOUT):
        return []
    qs = []
    for line in open(HOLDOUT, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            qs.append({"id": "h-%s-%s" % (d.get("doc"), d.get("page")), "q": d["query"],
                       "gold": {str(d["page"])}, "src": "留出集"})
    return qs


def encode(model, texts, batch=32):
    return model.encode(texts, batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)


def evaluate(model, corpus_emb, pages, qs, tag):
    """corpus_emb 预先算好（每个模型只编码一次全库，避免 4 次重复编码）。"""
    t0 = time.time()
    qe = encode(model, [x["q"] for x in qs], batch=16)
    enc_ms = (time.time() - t0) * 1000
    sims = qe @ corpus_emb.T
    h1 = h3 = h5 = 0
    rr = 0.0
    detail = []
    for i, it in enumerate(qs):
        order = np.argsort(-sims[i])[:5]
        top_pages = [pages[j] for j in order]
        rank = next((k for k, j in enumerate(order, 1) if pages[j] in it["gold"]), None)
        h1 += rank == 1
        h3 += bool(rank and rank <= 3)
        h5 += bool(rank and rank <= 5)
        rr += (1.0 / rank) if rank else 0.0
        detail.append({"id": it["id"], "question": it["q"], "top_pages": top_pages,
                       "gold": sorted(it["gold"]), "rank": rank})
    n = len(qs)
    return {"tag": tag, "n": n, "hit@1": h1 / n, "hit@3": h3 / n, "hit@5": h5 / n,
            "mrr": rr / n, "encode_ms": round(enc_ms), "detail": detail}


def main():
    from sentence_transformers import SentenceTransformer
    corpus, pages, _ = load_corpus()
    qs = load_questions()
    print("语料块 =", len(corpus), "| 题目 =", len(qs), flush=True)
    hold = load_holdout()
    print("业务题 = %d ｜ 留出集 = %d" % (len(qs), len(hold)), flush=True)

    def dump(partial):
        json.dump(partial, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if partial.get("base") and partial.get("ft") and partial.get("holdout_base") and partial.get("holdout_ft"):
            write_md(partial)

    res = {}
    for name, path in (("base", BASE), ("ft", FTDIR)):
        t0 = time.time()
        model = SentenceTransformer(path)
        model.max_seq_length = 256
        print("[%s] 编码全库 %d 块…" % (name, len(corpus)), flush=True)
        corpus_emb = encode(model, corpus)
        print("[%s] 全库编码完成 %.1f 分钟" % (name, (time.time() - t0) / 60), flush=True)
        res["holdout_" + name] = evaluate(model, corpus_emb, pages, hold, name) if hold else None
        res[name] = evaluate(model, corpus_emb, pages, qs, name)
        for key, val in (("base", res.get("base")), ("ft", res.get("ft")),
                         ("holdout_base", res.get("holdout_base")), ("holdout_ft", res.get("holdout_ft"))):
            if val:
                print("%-13s Hit@1=%.1f%% Hit@3=%.1f%% Hit@5=%.1f%% MRR=%.4f" %
                      (key, 100 * val["hit@1"], 100 * val["hit@3"], 100 * val["hit@5"], val["mrr"]), flush=True)
        dump(res)
    write_md(res)
    print("saved ->", OUT_MD, "|", OUT_JSON)


def write_md(res):
    r_base, r_ft = res["base"], res["ft"]
    h_base, h_ft = res.get("holdout_base"), res.get("holdout_ft")

    def pct(x):
        return "%.1f%%" % (100 * x)

    md = ["# 工单 11 · Embedding 模型微调前后检索效果对比", "",
          "> 工单编号：人工智能 NLP-RAG 项目-Embeddings 模型微调任务",
          "> 语料：招股说明书 1/2（1,587 个文本/表格块）；题集：招股书1 中文 10 题 + 招股书2 6 题 = **16 题**（另有留出集 %d 题）" % (h_base["n"] if h_base else 0),
          "> 检索方式：纯向量（余弦）取 top-5，比对金标准页；嵌入同为一台本机 CPU。", "",
          "## 一、总体指标", "",
          "| 模型 | Hit@1 | Hit@3 | Hit@5 | MRR | 查询编码耗时 |",
          "|---|---|---|---|---|---|",
          "| 微调前 %s | %s | %s | %s | %.4f | %.1f s |" % (BASE, pct(r_base["hit@1"]), pct(r_base["hit@3"]), pct(r_base["hit@5"]), r_base["mrr"], r_base["encode_ms"] / 1000),
          "| **微调后（领域微调）** | **%s** | **%s** | **%s** | **%.4f** | %.1f s |" % (pct(r_ft["hit@1"]), pct(r_ft["hit@3"]), pct(r_ft["hit@5"]), r_ft["mrr"], r_ft["encode_ms"] / 1000),
          "", "**变化**：Hit@1 %+.1f pp ｜ Hit@3 %+.1f pp ｜ Hit@5 %+.1f pp ｜ MRR %+.4f" %
          (100 * (r_ft["hit@1"] - r_base["hit@1"]), 100 * (r_ft["hit@3"] - r_base["hit@3"]),
           100 * (r_ft["hit@5"] - r_base["hit@5"]), r_ft["mrr"] - r_base["mrr"]), "",
          "## 二、逐题对比（命中排名）", "",
          "| 题号 | 问题 | 微调前 | 微调后 | 微调后 top-5 页 |", "|---|---|---|---|---|"]
    for x, y in zip(r_base["detail"], r_ft["detail"]):
        md.append("| %s | %s | %s | %s | %s |" %
                  (x["id"], x["question"][:36], x["rank"] or "MISS", y["rank"] or "MISS", y["top_pages"]))
    if h_base and h_ft:
        md += ["", "## 三、留出集（页面留出 + 全新问法，%d 题）——泛化能力" % h_base["n"], "",
               "| 模型 | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|",
               "| 微调前 %s | %s | %s | %s | %.4f |" % (BASE, pct(h_base["hit@1"]), pct(h_base["hit@3"]),
                                                 pct(h_base["hit@5"]), h_base["mrr"]),
               "| **微调后** | **%s** | **%s** | **%s** | **%.4f** |" % (pct(h_ft["hit@1"]), pct(h_ft["hit@3"]),
                                                              pct(h_ft["hit@5"]), h_ft["mrr"]),
               "", "**变化**：Hit@1 %+.1f pp ｜ Hit@3 %+.1f pp ｜ Hit@5 %+.1f pp ｜ MRR %+.4f" %
               (100 * (h_ft["hit@1"] - h_base["hit@1"]), 100 * (h_ft["hit@3"] - h_base["hit@3"]),
                100 * (h_ft["hit@5"] - h_base["hit@5"]), h_ft["mrr"] - h_base["mrr"]),
               "", "## 四、逐题明细（留出集）", "", "| 题号 | 问题 | 微调前 | 微调后 | 微调后 top-5 页 |",
               "|---|---|---|---|---|"]
        for x, y in zip(h_base["detail"], h_ft["detail"]):
            md.append("| %s | %s | %s | %s | %s |" %
                      (x["id"].replace("h-招股说明书", ""), x["question"][:34],
                       x["rank"] or "MISS", y["rank"] or "MISS", y["top_pages"]))
    md += ["", "> 明细数据：`embedding_ft/评估_微调前后.json`；留出集：`embedding_ft/data/eval_pairs.jsonl`", ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("saved ->", OUT_MD)


if __name__ == "__main__":
    main()
