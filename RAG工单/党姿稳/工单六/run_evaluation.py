# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
评估脚本：在验收问题上对比三种检索策略（向量 / 全文 / 混合）的
  准确率（Top-K 是否含答案）与召回率（Top-N 召回集是否含答案），
  并对比不同融合算法、不同重排算法的影响，验证混合检索的价值。
"""
import os
import sys
import json
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from kb import load_all_chunks
from hybrid_retriever import HybridRetriever
from rerankers import TFIDFReranker
from rag_chain import judge_contexts
from config import EVAL_QUESTIONS, RECALL_TOP_N, ACCURACY_TARGET, RECALL_TARGET

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 对比配置：(策略, 重排器, 融合算法, 向量权重)
CONFIGS = [
    ("vector", "none", "-", 1.0),
    ("vector", "tfidf", "-", 1.0),
    ("vector", "llm", "-", 1.0),
    ("fulltext", "-", "-", 0.0),
    ("hybrid", "tfidf", "weighted", 0.6),
    ("hybrid", "llm", "weighted", 0.6),
    ("hybrid", "llm", "vote", 0.6),
]


def run_config(allc, name, strategy, reranker, fusion, vw):
    """跑一组配置，返回 {accuracy, recall, details}"""
    r = HybridRetriever(allc, reranker=reranker if reranker != "none" else "none",
                        vector_weight=vw, fulltext_weight=round(1 - vw, 2), fusion=fusion)
    hits, rec_items = [], []
    for item in EVAL_QUESTIONS:
        q = item["question"]
        if strategy == "fulltext":
            res = r.fulltext_search(q, top_k=5)
            broad = r.fulltext_search(q, top_k=RECALL_TOP_N)
        elif strategy == "vector":
            res = r.vector_search(q, top_k=5)
            broad = [(r.chunks[i], s) for i, s in r.vector_recall(q, RECALL_TOP_N)]
        else:
            res = r.hybrid_search(q, top_k=5, fusion=fusion, vector_weight=vw,
                                  fulltext_weight=round(1 - vw, 2))
            broad = r.hybrid_search(q, top_k=RECALL_TOP_N, fusion=fusion, vector_weight=vw,
                                    fulltext_weight=round(1 - vw, 2))
        h = judge_contexts(q, res)
        rc = judge_contexts(q, broad)
        hits.append(h)
        rec_items.append(rc)
        print(f"    Q{item['id']}: 准确={h} 召回={rc} 页{[c['page'] for c,_ in res]}")
    n = len(EVAL_QUESTIONS)
    return {"name": name, "accuracy": sum(hits) / n, "recall": sum(rec_items) / n,
            "details": [{"id": it["id"], "question": it["question"], "hit": h, "recalled": rc}
                        for it, h, rc in zip(EVAL_QUESTIONS, hits, rec_items)]}


def main():
    allc, _, _ = load_all_chunks()
    print(f"知识库：{len(allc)} 个文档块；验收问题 {len(EVAL_QUESTIONS)} 个")
    t0 = time.time()
    results = []
    for strategy, reranker, fusion, vw in CONFIGS:
        name = f"{strategy}+{reranker}+{fusion}"
        print(f"\n[配置] {name} ...")
        results.append(run_config(allc, name, strategy, reranker, fusion, vw))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    best = max(results, key=lambda r: (r["accuracy"] + r["recall"]))
    summary = {"timestamp": ts, "n_questions": len(EVAL_QUESTIONS),
               "elapsed": round(time.time() - t0, 1),
               "best_config": best["name"],
               "best_accuracy": round(best["accuracy"], 4),
               "best_recall": round(best["recall"], 4),
               "accuracy_target": ACCURACY_TARGET, "recall_target": RECALL_TARGET,
               "accuracy_target_met": best["accuracy"] >= ACCURACY_TARGET,
               "recall_target_met": best["recall"] >= RECALL_TARGET,
               "results": [{k: v for k, v in r.items() if k != "details"} for r in results]}

    json_path = os.path.join(OUTPUT_DIR, f"hybrid_eval_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(OUTPUT_DIR, f"hybrid_eval_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 工单六 混合检索 评估报告\n\n")
        f.write("**工单编号：** 人工智能NLP-RAG-混合检索任务\n\n")
        f.write(f"**评估时间：** {datetime.now():%Y-%m-%d %H:%M:%S}　**问题数：** {len(EVAL_QUESTIONS)}\n\n")
        f.write("## 各检索策略对比\n\n")
        f.write("| 检索配置（策略+重排+融合） | 准确率 Top-5 | 召回率 Top-20 |\n|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['accuracy']:.1%} | {r['recall']:.1%} |\n")
        f.write(f"\n**目标准确率 ≥{ACCURACY_TARGET:.0%}：** "
                f"{'达成 ✅' if summary['accuracy_target_met'] else '未达成 ❌'}（最优 {best['accuracy']:.1%}，配置 {best['name']}）\n\n")
        f.write(f"**目标召回率 ≥{RECALL_TARGET:.0%}：** "
                f"{'达成 ✅' if summary['recall_target_met'] else '未达成 ❌'}（最优 {best['recall']:.1%}）\n\n")
        f.write("## 最优配置逐题结果\n\n| 问题ID | 问题 | 准确命中 | 召回命中 |\n|---|---|---|---|\n")
        for d in best["details"]:
            f.write(f"| {d['id']} | {d['question'][:50]} | {'✅' if d['hit'] else '❌'} | "
                    f"{'✅' if d['recalled'] else '❌'} |\n")
    print(f"\n✅ 报告已保存:\n{json_path}\n{md_path}")
    print(f"最优配置 {best['name']}: 准确率 {best['accuracy']:.1%} / 召回率 {best['recall']:.1%}")


if __name__ == "__main__":
    main()
