# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：优化前后对比分析（工单产出物：01/02/03/04/05 系统 vs 06 系统）

对比口径分两层，**同层内才可比**（这是本报告最重要的方法论约束）：

第 1 层 · 同环境实跑消融（本脚本现场跑，16 题同一批问题、同一台机器）
  · baseline_01_sim  —— 仿真工单01：仅向量召回、无精排、只看招股1、
                        且排除表格块/图像块/组织图块（= 上游 01 的能力面）；
  · baseline_03_sim  —— 仿真工单03：向量+全文混合召回 + Reranker 精排，
                        含表格块、不含图像块与组织图块；
  · full_06          —— 本工单：三策略全开（含图像块/组织图块 + 精排 + 可选融合）。

第 2 层 · 上游系统自报实测（只读引用，来自各工单自己落盘的评估产物）
  · 工单1/3/4/5 的 data/eval、docs/reports 中的指标；数值与口径在报告里逐条标注来源，
    不重算、不修饰 —— 若上游口径与本地消融不同，报告里显式写明"不可直接横比"。

用法
----
    python evaluate_compare.py                      # 全跑（含答案生成）
    python evaluate_compare.py --no-answer          # 只算检索指标（快）
    python evaluate_compare.py --report docs/05-优化前后对比分析.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402

from evaluate_retrieval import (answer_accuracy, cite_summary, corpus_stats,  # noqa: E402
                                score_hits)


def _upstream_arms() -> list[dict]:
    """上游对比臂定义（消融式仿真 + 说明）。"""
    return [
        {
            "key": "baseline_01_sim",
            "label": "工单01 仿真（单文档 · 纯向量 · 无精排 · 无表格/图像块）",
            "cfg": {"strategy": "vector", "rerank_enabled": False,
                    "sources": ["招股说明书1.pdf"],
                    "exclude_block_types": ["table_row", "table_header", "table",
                                            "image", "orgchart"]},
        },
        {
            "key": "baseline_03_sim",
            "label": "工单03 仿真（双文档 · 向量+全文融合 · Reranker 精排 · 含表格块）",
            "cfg": {"strategy": "hybrid", "rerank_enabled": True,
                    "exclude_block_types": ["image", "orgchart"]},
        },
        {
            "key": "full_06",
            "label": "工单06 本系统（三策略 · 表格+图像+组织图块 · Reranker 精排 + 融合）",
            "cfg": {"strategy": "hybrid", "rerank_enabled": True},
        },
    ]


def run_arms(strategies_mod, runtime_mod, rag_mod, arms, questions, gts, corpus_cache,
             k: int, with_answer: bool) -> list[dict]:
    rows: list[dict] = []
    for arm in arms:
        print(f"\n=== 对比臂 {arm['key']}：{arm['label']}", flush=True)
        # 【配置隔离】每个对比臂开始前先 reset，再写入本臂配置。
        # 不能只 update：update 会跳过值为 None 的字段，上一臂的 sources /
        # exclude_block_types 会**泄漏**到下一臂 —— 实测踩坑：baseline_03_sim 与
        # full_06 继承了 baseline_01_sim 的「只看招股1」限制，招股2 的 6 道题
        # 全部检索不到，信息召回率从 0.875 假摔到 0.608。
        runtime_mod.reset()
        runtime_mod.update(**dict(arm["cfg"]))
        cfg = runtime_mod.get()
        for qi, q in enumerate(questions, 1):
            gt = gts.get(q["id"])
            if not gt:
                continue
            t0 = time.time()
            res = strategies_mod.retrieve(q["question"], cfg)
            retrieval_ms = round((time.time() - t0) * 1000, 1)
            metrics = score_hits(res["hits"], gt, k, corpus_cache[q["id"]])
            row = {"arm": arm["key"], "id": q["id"], "question": q["question"],
                   "metrics": metrics, "retrieval_ms": retrieval_ms,
                   "rerank": (res["info"].get("rerank") or {}).get("algorithm"),
                   "n_candidates": res["info"].get("n_candidates")}
            if with_answer:
                t1 = time.time()
                # 答案生成必须与检索口径**完全一致**：把该对比臂的全部配置
                # （策略 / 精排开关 / 文档范围 / 块类型过滤）都写进运行时设置，
                # 否则 rag.ask 会绕过消融约束全库作答 —— 结论就不可比了
                # （首版就踩了这个坑：baseline_01 的检索召回只有 35%，
                #   答案准确率却和 full_06 一样高）。
                runtime_mod.update(**dict(arm["cfg"]))
                r = rag_mod.ask(q["question"], use_cache=False)
                row["answer"] = r.answer
                row["answer_accuracy"] = answer_accuracy(r.answer, gt)
                row["total_ms"] = (r.timings or {}).get("total_ms")
                row["answer_ms"] = round((time.time() - t1) * 1000, 1)
                row["citations"] = cite_summary(r.citations)
            rows.append(row)
            m = metrics
            print(f"  [{qi}/{len(questions)}] id={q['id']:>4} "
                  f"信息召回={m['fact_coverage@k']:.2f} P@{k}={m['precision@k']:.2f} "
                  f"MRR={m['mrr']:.2f} nDCG={m['ndcg@k']:.2f} "
                  f"({retrieval_ms:.0f}ms)", flush=True)
    return rows


def aggregate(rows: list[dict], arms: list[dict], k: int) -> dict:
    out: dict[str, dict] = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm["key"]]
        if not rs:
            continue
        n = len(rs)
        agg = {
            "label": arm["label"], "n": n,
            "fact_coverage@k": round(sum(r["metrics"]["fact_coverage@k"] for r in rs) / n, 4),
            "fact_all_covered_rate": round(
                sum(1 for r in rs if r["metrics"]["fact_all_covered@k"]) / n, 4),
            "hit_rate@k": round(sum(r["metrics"]["hit_rate@k"] for r in rs) / n, 4),
            "precision@k": round(sum(r["metrics"]["precision@k"] for r in rs) / n, 4),
            "recall@k": round(sum(r["metrics"]["recall@k"] for r in rs) / n, 4),
            "mrr": round(sum(r["metrics"]["mrr"] for r in rs) / n, 4),
            "ndcg@k": round(sum(r["metrics"]["ndcg@k"] for r in rs) / n, 4),
            "retrieval_ms_mean": round(sum(r["retrieval_ms"] for r in rs) / n, 1),
        }
        got = [r for r in rs if "answer_accuracy" in r]
        if got:
            agg["answer_accuracy"] = round(
                sum(r["answer_accuracy"]["coverage"] for r in got) / len(got), 4)
            agg["answer_all_hit_rate"] = round(
                sum(1 for r in got if r["answer_accuracy"]["all_hit"]) / len(got), 4)
            tt = [r["total_ms"] for r in got if r.get("total_ms")]
            if tt:
                agg["total_ms_mean"] = round(sum(tt) / len(tt), 1)
                agg["within_3s_rate"] = round(sum(1 for t in tt if t <= 3000) / len(tt), 4)
        out[arm["key"]] = agg
    return out


def collect_upstream() -> list[dict]:
    """只读收集上游工单自己落盘的实测指标（含来源路径，便于溯源）。"""
    up = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found: list[dict] = []

    def _load(path: str):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    # 工单5（多轮）验收结果
    d5 = _load(os.path.join(up, "工单5", "data", "eval", "acceptance_result.json"))
    if isinstance(d5, dict):
        found.append({
            "workorder": "工单05 Query理解优化（多轮对话）",
            "source": "工单5/data/eval/acceptance_result.json",
            "metrics": {"n_turns": d5.get("n_turns"), "n_failed": d5.get("n_failed"),
                        "max_ms": max([r.get("total_ms", 0) for r in d5.get("rows", [])] or [0]),
                        "all_passed": d5.get("all_passed")},
        })
    # 工单4（图像解析）检索精确度
    d4 = _load(os.path.join(up, "工单4", "data", "eval", "retrieval_metrics_full_04.json"))
    if isinstance(d4, dict):
        found.append({"workorder": "工单04 图像内容解析",
                      "source": "工单4/data/eval/retrieval_metrics_full_04.json",
                      "metrics": d4})
    # 工单3 对比结果
    d3 = _load(os.path.join(up, "工单3", "data", "eval", "compare_results.json"))
    if isinstance(d3, dict):
        found.append({"workorder": "工单03 表格解析与检索优化",
                      "source": "工单3/data/eval/compare_results.json",
                      "metrics": d3.get("aggregate") or d3.get("summary") or {}})
    return found


def render_report(result: dict) -> str:
    k = result["k"]
    L: list[str] = []
    A = L.append
    A("# 工单06 优化前后对比分析")
    A("")
    A("> **工单编号**：人工智能NLP-RAG-混合检索任务  ")
    A(f"> **实跑环境**：本机（Windows 11 / RTX 4060 Laptop 8GB / Python 3.12 / "
      f"bge-m3 + bge-reranker-v2-m3）  ")
    A(f"> **同批问题**：16 题（`data/test_questions.json`），k={k}  ")
    A(f"> **总耗时**：{result['elapsed_s']}s")
    A("")
    A("---")
    A("")
    A("## 1. 同环境实跑对比（消融式，可比性最强）")
    A("")
    A("同代码、同语料、同模型、同一批题，只切换**能力开关**。")
    A("这是本报告的主要证据：三个对比臂之间只有策略/块类型/精排的差异，其他变量全同。")
    A("")
    A(f"| 对比臂 | 信息召回率@{k} | 全召回率 | Hit Rate@{k} | Precision@{k} | "
      f"Recall@{k} | MRR | NDCG@{k} | 答案准确率 | 答案全对率 | 端到端均值 | ≤3s 占比 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, a in result["aggregate"].items():
        A(f"| {a['label']} | **{a['fact_coverage@k']:.4f}** | {a['fact_all_covered_rate']:.4f} | "
          f"{a['hit_rate@k']:.4f} | {a['precision@k']:.4f} | {a['recall@k']:.4f} | "
          f"{a['mrr']:.4f} | {a['ndcg@k']:.4f} | {a.get('answer_accuracy', '-')} | "
          f"{a.get('answer_all_hit_rate', '-')} | {a.get('total_ms_mean', '-')}ms | "
          f"{a.get('within_3s_rate', '-')} |")
    A("")
    if "baseline_01_sim" in result["aggregate"] and "full_06" in result["aggregate"]:
        b = result["aggregate"]["baseline_01_sim"]
        f = result["aggregate"]["full_06"]
        A("**提升（06 vs 01 仿真）**：")
        A("")
        for name, key2 in (("信息召回率", "fact_coverage@k"), ("Hit Rate", "hit_rate@k"),
                           ("MRR", "mrr"), ("NDCG", "ndcg@k"),
                           ("答案准确率", "answer_accuracy")):
            if b.get(key2) is not None and f.get(key2) is not None:
                A(f"- {name}：{b[key2]:.4f} → {f[key2]:.4f}"
                  f"（{'↑' if f[key2] >= b[key2] else '↓'} "
                  f"{abs(f[key2] - b[key2]):.4f}）")
        A("")
    if result.get("upstream"):
        A("---")
        A("")
        A("## 2. 上游系统自报实测（只读引用，口径可能不同，**不可直接横比**）")
        A("")
        A("| 上游工单 | 来源文件 | 关键指标 |")
        A("| --- | --- | --- |")
        for u in result["upstream"]:
            m = json.dumps(u["metrics"], ensure_ascii=False)
            A(f"| {u['workorder']} | `{u['source']}` | {m[:400]} |")
        A("")
        A("> 说明：各工单的评估集、口径、机器时点并不完全一致（如工单4 的 16 题评估包含"
          "其独有的图像链路与直接抽取评分），因此本节只作**历史归档**，"
          "真正的结论请以第 1 节的同环境消融为准。")
    A("")
    A("---")
    A("")
    A("## 3. 逐题明细")
    A("")
    for key, a in result["aggregate"].items():
        A(f"### {a['label']}")
        A("")
        A(f"| 题号 | 信息召回率 | 命中事实数 | Hit | MRR | NDCG | 检索耗时 |")
        A("| --- | --- | --- | --- | --- | --- | --- |")
        for r in [x for x in result["rows"] if x["arm"] == key]:
            m = r["metrics"]
            A(f"| {r['id']} | {m['fact_coverage@k']:.2f} | "
              f"{len(m['facts_covered'])}/{m['facts_total']} | {m['hit_rate@k']:.0f} | "
              f"{m['mrr']:.2f} | {m['ndcg@k']:.2f} | {r['retrieval_ms']:.0f}ms |")
        A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单06 优化前后对比分析")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-answer", action="store_true", help="跳过 LLM 生成（只比检索）")
    ap.add_argument("--out", default=os.path.join("data", "eval", "compare_06.json"))
    ap.add_argument("--report", default=os.path.join("docs", "05-优化前后对比分析.md"))
    args = ap.parse_args()

    from src import fulltext, rag, runtime, strategies

    arms = _upstream_arms()
    gt_all = json.load(open(os.path.join("data", "eval", "ground_truth.json"),
                            encoding="utf-8"))
    gts = {g["id"]: g for g in gt_all}
    questions = json.load(open(os.path.join("data", "test_questions.json"),
                               encoding="utf-8"))

    print("[compare] 预热……", flush=True)
    rag.warmup()
    payloads = fulltext.build().payloads
    corpus_cache = {qid: corpus_stats(gts[qid], payloads)
                    for qid in (q["id"] for q in questions) if qid in gts}

    t0 = time.time()
    rows = run_arms(strategies, runtime, rag, arms, questions, gts, corpus_cache,
                    args.top_k, not args.no_answer)
    result = {
        "workorder": "人工智能NLP-RAG-混合检索任务",
        "k": args.top_k,
        "n_questions": len(questions),
        "with_answer": not args.no_answer,
        "aggregate": aggregate(rows, arms, args.top_k),
        "rows": rows,
        "upstream": collect_upstream(),
        "elapsed_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(render_report(result))
    print(f"\n[compare] 完成：{args.out} / {args.report}")
    for key, a in result["aggregate"].items():
        print(f"  {key:<16} 信息召回={a['fact_coverage@k']:.3f} "
              f"Hit={a['hit_rate@k']:.3f} MRR={a['mrr']:.3f} "
              f"答案准确率={a.get('answer_accuracy', '-')}")
    return 0


if __name__ == "__main__":
    bootstrap.run_with_large_stack(main)
