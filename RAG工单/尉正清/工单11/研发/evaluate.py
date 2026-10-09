# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
"""评估器入口：在固定评估集上算检索指标

微调前后各跑一次，两次的差异就是微调带来的收益。

用法：
    python evaluate.py --model base        # 微调前
    python evaluate.py --model finetuned   # 微调后
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer

import data as D
import retrieval as R
from config import BASE_MODEL, FINETUNED_MODEL

OUT = D.DATA_DIR
RESULTS = Path(__file__).parent.parent / "测试" / "results"


def load_inputs():
    eval_set = json.loads((OUT / "eval_set.json").read_text(encoding="utf-8"))
    corpus = D._read_jsonl(OUT / "corpus.jsonl")
    return eval_set, corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "finetuned"], required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--tag", default=None, help="结果文件名后缀")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    model_path = Path(args.model_path) if args.model_path else (
        BASE_MODEL if args.model == "base" else FINETUNED_MODEL)
    if not model_path.exists():
        print(f"[错误] 模型不存在：{model_path}")
        if args.model == "finetuned":
            print("       先跑 python finetune.py 训练")
        return 1

    tag = args.tag or args.model
    print("=" * 68)
    print(f"评估：{tag}  ({model_path})")
    print("=" * 68)
    print(R.gpu_report())

    eval_set, corpus = load_inputs()
    rel = {q: set(ids) for q, ids in eval_set["qrels"].items()}
    queries = eval_set["queries"]
    print(f"\n评估集：{len(queries)} 个 query | 语料 {len(corpus):,} 篇")

    model = SentenceTransformer(str(model_path))

    print("\n编码语料 ...")
    salt = R.model_salt(model_path)      # 模型换了，缓存自动失效
    corpus_emb = R.encode(model, [c["text"] for c in corpus], tag,
                          cache_key=R._fingerprint(model, corpus, salt))
    print("编码查询 ...")
    # 查询侧加 BGE 指令前缀（微调前后同一口径，见 config.QUERY_INSTRUCTION）
    q_emb = R.encode(model, [D.with_instruction(q["text"]) for q in queries], tag,
                     cache_key=None, show=False)

    print(f"\n检索 top-{args.top_k} 并计算指标 ...")
    idx, _ = R.search(q_emb, corpus_emb, top_k=args.top_k)
    corpus_ids = [c["_id"] for c in corpus]
    ranked = {q["_id"]: [corpus_ids[i] for i in row]
              for q, row in zip(queries, idx)}

    metrics = R.evaluate(ranked, rel)

    print("\n" + "-" * 68)
    for name in sorted(metrics, key=lambda n: (n.split("@")[0], int(n.split("@")[1]))):
        print(f"  {name:<12} {metrics[name]:.4f}")
    print("-" * 68)

    # 逐 query 明细：报告里要能看出"提升集中在哪类问题上"，而不只是一个总分
    detail = []
    for q in queries:
        gold = rel[q["_id"]]
        top10 = ranked[q["_id"]][:10]
        detail.append({
            "query_id": q["_id"], "query": q["text"],
            "n_gold": len(gold),
            "hit@10": len(set(top10) & gold),
            "first_rank": next((i + 1 for i, d in enumerate(ranked[q["_id"]])
                                if d in gold), None),
        })

    R.save_result(RESULTS / f"{tag}.json", {
        "tag": tag, "model_path": str(model_path),
        "n_eval_queries": len(queries), "corpus_size": len(corpus),
        "top_k": args.top_k, "metrics": metrics, "per_query": detail,
    })
    print(f"\n结果已写入 {RESULTS / f'{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
