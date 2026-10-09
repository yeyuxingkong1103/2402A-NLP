# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""第一步：生成微调数据集

产出三份文件（都在 cache/fiqa/ 下）：
  corpus.jsonl        语料快照（57638 篇）
  train_triples.jsonl 训练三元组 (anchor, positive, negative)
  eval_set.json       评估集：query + 标准答案 + 语料快照

用法：python build_dataset.py [--force]
"""
import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer

import data as D
import retrieval as R
from config import BASE_MODEL, HARD_NEG_PER_QUERY, HARD_NEG_TOP_K

OUT = D.DATA_DIR
BASE_TAG = "base"


def mine_hard_negatives(model, corpus, queries, rel, train_qids):
    """给每个训练 query 挖难负例。

    做法：用**微调前**的模型检索整个语料，取排名靠前但不相关的文档。
    为什么不用随机负例：随机文档跟查询八竿子打不着，模型不用学就能分开；
    真正难的是"字面很像但答案不对"的那些，金融领域尤其如此
    （比如问 A 公司的营收，检索到 B 公司格式一模一样的财报段落）。
    """
    corpus_ids = [c["_id"] for c in corpus]
    corpus_emb = R.encode(model, [c["text"] for c in corpus], BASE_TAG,
                          cache_key=R._fingerprint(
                              model, corpus, R.model_salt(BASE_MODEL)))

    qtext = {q["_id"]: q["text"] for q in queries}
    train_queries = [{"_id": q, "text": qtext[q]} for q in train_qids if q in qtext]
    # 挖负例时也要带指令前缀，否则挖出来的负例和训练时的查询分布对不上
    q_emb = R.encode(model, [D.with_instruction(q["text"]) for q in train_queries],
                     BASE_TAG, cache_key=None, show=False)

    idx, _ = R.search(q_emb, corpus_emb, top_k=HARD_NEG_TOP_K)

    hard = {}
    for qi, row in zip(train_queries, idx):
        gold = rel[qi["_id"]]
        picked = [corpus_ids[i] for i in row if corpus_ids[i] not in gold]
        hard[qi["_id"]] = picked[:HARD_NEG_PER_QUERY]
    return hard, corpus_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="重新下载并重建")
    args = ap.parse_args()

    print("=" * 68)
    print("第一步：生成微调数据集")
    print("=" * 68)
    print(R.gpu_report())

    corpus, queries, qrels = D.load_fiqa(force=args.force)
    rel = D.relevance_map(qrels)
    train_qids, eval_qids = D.split_queries(rel)
    print(f"\n语料 {len(corpus):,} 篇 | query {len(queries):,} 条 | "
          f"有标注 {len(rel):,} 条")
    print(f"切分：训练 {len(train_qids)} 个 query / 评估 {len(eval_qids)} 个 query")
    print(f"      （按 query 切分，避免同一问题的正例同时出现在两边）")

    print(f"\n加载基座模型 {BASE_MODEL.name} ...")
    model = SentenceTransformer(str(BASE_MODEL))

    print(f"\n挖掘难负例（每 query 从 top-{HARD_NEG_TOP_K} 里挑 "
          f"{HARD_NEG_PER_QUERY} 个不相关的）...")
    hard, corpus_emb = mine_hard_negatives(model, corpus, queries, rel, train_qids)

    triples = D.build_triples({q: rel[q] for q in train_qids}, queries, hard)
    print(f"  → 生成三元组 {len(triples):,} 条")

    # 落盘
    D.write_jsonl(OUT / "train_triples.jsonl", triples)
    D.write_jsonl(OUT / "corpus.jsonl", corpus)

    eval_set = {
        "queries": [{"_id": q, "text": next(x["text"] for x in queries
                                            if x["_id"] == q)} for q in eval_qids],
        "qrels": {q: sorted(rel[q]) for q in eval_qids},
        "corpus_size": len(corpus),
    }
    (OUT / "eval_set.json").write_text(
        json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")

    # 抽几条给人看，交付物里需要能直观检验数据集质量
    with open(OUT / "dataset_preview.txt", "w", encoding="utf-8") as fh:
        ctext = {c["_id"]: c["text"] for c in corpus}
        for t in triples[:5]:
            fh.write(f"QUERY   : {t['anchor']}\n")
            fh.write(f"POSITIVE: {ctext[t['positive_id']][:200]}\n")
            fh.write(f"NEGATIVE: {ctext[t['negative_id']][:200]}\n")
            fh.write("-" * 68 + "\n")

    print(f"\n数据集已写入 {OUT}")
    print(f"  train_triples.jsonl  {len(triples):,} 条")
    print(f"  eval_set.json        {len(eval_qids)} 个 query")
    print(f"  corpus.jsonl         {len(corpus):,} 篇")
    return 0


if __name__ == "__main__":
    sys.exit(main())
