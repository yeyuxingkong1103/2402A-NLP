# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
工单07 · 检索与生成指标汇总（需求硬性要求4）

指标口径**直接复用**上游 06 的 `evaluate_retrieval.py`
（`score_hits` / `corpus_stats` / `answer_accuracy` / `ndcg_at_k` 原样 import，
不另写一套，保证与 01~06 报告可比）：

  检索指标：Precision@k、Recall@k、Hit Rate、MRR、NDCG@k
  生成指标：Answer Accuracy（关键事实覆盖率）、事实覆盖事实清单

与上游 evaluate_retrieval.py 的唯一差别在**数据来源**：
本脚本消费 `run_rag_questions.py` 产出的 `rag_results_<strategy>.json`。
那份留痕走的是 `rag.retrieve`（含 Query 理解 + **文档路由**），与 `rag.ask`
生成答案时是同一条链路；而上游 evaluate_retrieval.py 直接调 `strategies.retrieve`
裸检索，会跳过 doc_router —— 在本工单这种「9 份年报、大量同构报表」的语料上，
跳过路由会让他公司同名表格挤占 top-k，指标失真（实测 hybrid 下 id6 检索覆盖
从 0.00 变 1.00）。故此处以"系统真实链路"为准。

用法
----
    python evaluate_rag07.py                       # 汇总全部已有策略结果
    python evaluate_rag07.py --strategies hybrid
    python evaluate_rag07.py --top-k 5

输出
----
    data/eval/metrics_07.json      机器可读
    docs/05-RAG评估结果.md          人读逐题指标表 + 汇总表
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(strategy: str) -> dict | None:
    p = os.path.join(ROOT, "data", "eval", f"rag_results_{strategy}.json")
    if not os.path.isfile(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="工单07 指标汇总")
    ap.add_argument("--strategies", default="hybrid,vector,fulltext")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("data", "eval", "metrics_07.json"))
    ap.add_argument("--report", default=os.path.join("docs", "05-RAG评估结果.md"))
    args = ap.parse_args()

    # 复用上游的指标实现与相关性口径
    import evaluate_retrieval as ev
    from src import fulltext

    gt_all = json.load(open(os.path.join(ROOT, "data", "eval", "ground_truth.json"),
                            encoding="utf-8"))
    gts = {g["id"]: g for g in gt_all}

    print("[eval07] 载入全文索引以统计全语料相关块（Recall 分母）……", flush=True)
    payloads = fulltext.build().payloads
    corpus = {qid: ev.corpus_stats(g, payloads) for qid, g in gts.items()}
    print(f"[eval07] 语料 {len(payloads)} 块；金标 {len(gts)} 题", flush=True)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    all_rows: list[dict] = []
    agg: dict[str, dict] = {}

    for strat in strategies:
        data = _load(strat)
        if not data:
            print(f"[eval07] 跳过 {strat}（缺 data/eval/rag_results_{strat}.json）")
            continue
        for r in data["rows"]:
            gt = gts.get(r["id"])
            if not gt:
                continue
            m = ev.score_hits(r["hits"], gt, args.top_k, corpus[r["id"]])
            aa = ev.answer_accuracy(r.get("answer") or "", gt) \
                if r.get("answer") is not None else {}
            all_rows.append({
                "id": r["id"], "strategy": strat, "qtype": r.get("qtype"),
                "question": r["question"],
                "metrics": m,
                "answer_accuracy": aa,
                "answer": r.get("answer"),
                "retrieval_ms": r.get("retrieval_ms"),
                "timings": r.get("timings"),
                "doc_route": r.get("doc_route"),
            })

    # 汇总（按策略平均）
    for strat in strategies:
        rs = [r for r in all_rows if r["strategy"] == strat]
        if not rs:
            continue
        n = len(rs)
        keys = ("precision@k", "recall@k", "fact_coverage@k", "hit_rate@k", "mrr", "ndcg@k")
        a = {k: round(sum(r["metrics"][k] for r in rs) / n, 4) for k in keys}
        accs = [r["answer_accuracy"].get("coverage") for r in rs
                if r["answer_accuracy"].get("coverage") is not None]
        a["answer_accuracy"] = round(sum(accs) / len(accs), 4) if accs else None
        a["n_questions"] = n
        a["avg_retrieval_ms"] = round(
            sum(r["retrieval_ms"] or 0 for r in rs) / n, 1)
        agg[strat] = a

    result = {"top_k": args.top_k, "aggregate": agg, "rows": all_rows}
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    with open(os.path.join(ROOT, args.report), "w", encoding="utf-8") as fh:
        fh.write(render(result, strategies))

    print(f"\n[eval07] 完成：{args.out} / {args.report}")
    k = args.top_k
    hdr = f"  {'策略':<9}{'Hit@'+str(k):>9}{'P@'+str(k):>9}{'R@'+str(k):>9}" \
          f"{'MRR':>9}{'nDCG@'+str(k):>10}{'事实覆盖':>10}{'答案准确率':>11}"
    print(hdr)
    for s, a in agg.items():
        print(f"  {s:<9}{a['hit_rate@k']:>9.3f}{a['precision@k']:>9.3f}"
              f"{a['recall@k']:>9.3f}{a['mrr']:>9.3f}{a['ndcg@k']:>10.3f}"
              f"{a['fact_coverage@k']:>10.3f}"
              f"{(a['answer_accuracy'] if a['answer_accuracy'] is not None else float('nan')):>11.3f}")
    return 0


def render(result: dict, strategies: list[str]) -> str:
    k = result["top_k"]
    L: list[str] = []
    A = L.append
    A("# RAG 评估结果（工单07 · 需求硬性要求4）\n")
    A("> **工单编号**：人工智能NLP-RAG-功能测试及评估  ")
    A(f"> **评估口径**：复用上游 06 `evaluate_retrieval.py` 的指标实现"
      f"（`score_hits`/`corpus_stats`/`answer_accuracy`/`ndcg_at_k`）  ")
    A(f"> **k = {k}**　**策略**：{', '.join(strategies)}\n")

    A("## 一、指标定义（本报告口径）\n")
    A("| 指标 | 定义 |")
    A("| --- | --- |")
    A(f"| Precision@{k} | top-{k} 中相关块占比。相关 = 该块正文含本题任一 `evidence_facts` |")
    A(f"| Recall@{k} | top-{k} 相关块数 ÷ **全语料**相关块数（严格 IR 口径，分母大，天然偏低） |")
    A(f"| Hit Rate@{k} | top-{k} 是否至少含 1 个相关块（0/1，再按题平均） |")
    A("| MRR | 第一个相关块名次的倒数（1/rank） |")
    A(f"| NDCG@{k} | 以相关/不相关为增益的二值 nDCG |")
    A("| 事实覆盖@k | top-k 上下文并集覆盖本题关键事实的比例（**信息召回率**，业务口径） |")
    A("| Answer Accuracy | 生成答案对 `key_facts` 的逐字覆盖率 |")
    A("")
    A("> 说明：`Recall@k` 的低值主要来自**分母**——同一事实在年报里会以正文、多处表格、")
    A("> 附注等形式重复出现，形成几十个相关块；该指标用于**横向比较策略**，不宜单独解读。")
    A("> 业务上更应看**事实覆盖@k**与**答案准确率**。\n")

    A("## 二、策略汇总对比\n")
    A(f"| 策略 | Hit@{k} | Precision@{k} | Recall@{k} | MRR | NDCG@{k} | 事实覆盖@{k} | Answer Accuracy | 平均检索耗时 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in strategies:
        a = result["aggregate"].get(s)
        if not a:
            continue
        aa = a["answer_accuracy"]
        A(f"| **{s}** | {a['hit_rate@k']:.3f} | {a['precision@k']:.3f} | "
          f"{a['recall@k']:.3f} | {a['mrr']:.3f} | {a['ndcg@k']:.3f} | "
          f"{a['fact_coverage@k']:.3f} | "
          f"{(f'{aa:.3f}' if aa is not None else '-')} | {a['avg_retrieval_ms']} ms |")
    A("")

    A(f"## 三、逐题指标\n")
    for s in strategies:
        rs = [r for r in result["rows"] if r["strategy"] == s]
        if not rs:
            continue
        A(f"### 策略 `{s}`\n")
        A(f"| 题号 | 题型 | 检索 | Hit@{k} | P@{k} | R@{k} | MRR | NDCG@{k} | 事实覆盖 | 答案准确率 | 检索耗时 |")
        A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in rs:
            m = r["metrics"]
            acc = r["answer_accuracy"].get("coverage")
            A(f"| {r['id']} | {r.get('qtype') or '-'} | {r['question'][:26]}… | "
              f"{m['hit_rate@k']:.2f} | {m['precision@k']:.2f} | {m['recall@k']:.2f} | "
              f"{m['mrr']:.2f} | {m['ndcg@k']:.2f} | {m['fact_coverage@k']:.2f} | "
              f"{(f'{acc:.2f}' if acc is not None else '-')} | "
              f"{r['retrieval_ms']} ms |")
        A("")

    A("## 四、逐题关键事实命中明细\n")
    for s in strategies:
        rs = [r for r in result["rows"] if r["strategy"] == s]
        if not rs:
            continue
        A(f"### 策略 `{s}`\n")
        for r in rs:
            m = r["metrics"]
            A(f"- **id={r['id']}**（{r.get('qtype')}）"
              f"检索命中 {m['facts_covered']}；遗漏 {m['facts_missed']}")
            if r["answer_accuracy"]:
                A(f"  - 答案遗漏 {r['answer_accuracy'].get('miss') or '无'}")
        A("")
    return "\n".join(L)


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
