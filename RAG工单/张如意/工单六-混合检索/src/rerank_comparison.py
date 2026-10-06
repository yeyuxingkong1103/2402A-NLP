# -*- coding: utf-8 -*-
"""
工单06 三种重排算法对比实验（none / tfidf / llm / adaptive / cascade）
工单编号：人工智能NLP-RAG-混合检索任务

实验设计（控制变量）：
  · 检索策略固定为 hybrid + rrf 融合（工单06 推荐配置），召回 Top-20；
  · 仅切换重排器，比较最终 Top-5 的检索质量与耗时；
  · 问题集为工单问题（兴图新科 10 问 + 力源信息 4 问），
    关键信息点 anchors 已对照招股书原文核实（见 wo06_common.py）；
  · 自适应重排器使用独立的反馈文件，保证实验可复现、不污染线上反馈。

指标：Hit Rate@5 / MRR / Recall@k（关键信息点召回）/ 准确率 / 平均耗时。

运行：
    python 工单06-混合检索/src/rerank_comparison.py
产出：
    工单06-混合检索/results/rerank_comparison.json
    工单06-混合检索/results/rerank_comparison.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_core import config, rerank as rerank_mod            # noqa: E402
from wo06_common import (ACC_TARGET, RECALL_K, REC_RATIO_TARGET, RESULTS_DIR,  # noqa: E402
                         TOP_K, WO06_QA_SET, evaluate, md_table,
                         passage_lines, print_metrics, save_json, save_markdown)

RERANKERS = ["none", "tfidf", "llm", "adaptive", "cascade"]
FUSION = "rrf"
STRATEGY = "hybrid"


def run_one(retr, name: str) -> tuple[dict[int, list[dict]], dict[int, float],
                                      dict[int, tuple[int, int, bool]], list[str]]:
    """跑完整个问题集，返回 {qid: docs} / {qid: 耗时} / {qid: 覆盖统计} / 错误列表。"""
    docs_by_qid: dict[int, list[dict]] = {}
    latencies: dict[int, float] = {}
    qstats: dict[int, tuple[int, int, bool]] = {}
    errors: list[str] = []
    for q in WO06_QA_SET:
        try:
            res = retr.retrieve(q.question, strategy=STRATEGY, fusion=FUSION,
                                reranker=name, top_k=TOP_K, recall_k=RECALL_K)
            docs_by_qid[q.qid] = res.docs
            latencies[q.qid] = res.timings.get("total", 0.0)
        except Exception as e:                       # 单个问题失败不影响整体实验
            docs_by_qid[q.qid] = []
            latencies[q.qid] = 0.0
            errors.append(f"{q.qid}: {type(e).__name__}: {e}")
        docs = docs_by_qid[q.qid]
        cov = [any((a in (d.get("text") or "")) or (a in (d.get("section") or ""))
                   for d in docs) for a in q.anchors]
        qstats[q.qid] = (sum(cov), len(cov), bool(cov) and all(cov))
    return docs_by_qid, latencies, qstats, errors


def per_question_rows(qstats: dict[int, tuple[int, int, bool]],
                      latencies: dict[int, float]) -> list[list[str]]:
    """逐问题明细：关键信息点覆盖 / 是否可答对 / 耗时。"""
    rows = []
    for q in WO06_QA_SET:
        n_hit, n_all, passed = qstats.get(q.qid, (0, len(q.anchors), False))
        rows.append([q.qid, q.question[:34] + "…", f"{n_hit}/{n_all}",
                     "是" if passed else "否", f"{latencies.get(q.qid, 0):.2f}"])
    return rows


def main() -> None:
    # ---- 隔离自适应重排器的反馈文件，保证实验可复现 ----
    demo_feedback = RESULTS_DIR / "rerank_comparison_feedback.json"
    if demo_feedback.exists():
        demo_feedback.unlink()
    original_feedback = rerank_mod.FEEDBACK_FILE
    rerank_mod.FEEDBACK_FILE = demo_feedback

    try:
        from build_index import DEFAULT_MODEL, ensure_index, get_retriever
        ensure_index(verbose=True)
        retr = get_retriever(DEFAULT_MODEL)

        print("=" * 72)
        print(f"工单06 重排算法对比 | 策略={STRATEGY}/{FUSION}，"
              f"召回 Top-{RECALL_K} -> 输出 Top-{TOP_K}")
        print("=" * 72)
        if not config.DEEPSEEK_API_KEY:
            print("[提示] 未配置 DEEPSEEK_API_KEY，llm/cascade 将自动回退"
                  "（脚本仍可完整运行，配置后复跑可得真实 LLM 重排效果）。")

        summary_rows, details, all_metrics, all_qstats = [], {}, {}, {}
        for name in RERANKERS:
            docs_by_qid, latencies, qstats, errors = run_one(retr, name)
            all_qstats[name] = (qstats, latencies)
            m = evaluate(docs_by_qid, latencies, k=TOP_K)
            all_metrics[name] = m
            print_metrics(f"{name:8s}", m)
            if errors:
                print(f"  [warn] {len(errors)} 个问题异常：{errors[:2]}")
            details[name] = {
                str(q.qid): [
                    {"chunk_id": d.get("chunk_id"), "doc": d.get("doc"),
                     "page": d.get("page"), "type": d.get("type"),
                     "score": round(float(d.get("final_score", d.get("score", 0))), 4),
                     "snippet": (d.get("text") or "")[:120]}
                    for d in docs_by_qid.get(q.qid, [])[:TOP_K]
                ] for q in WO06_QA_SET
            }
            summary_rows.append([
                name,
                f"{m['accuracy']:.2%}", f"{m['recall']:.2%}",
                f"{m['hit_rate']:.2%}", f"{m['mrr']:.4f}",
                f"{m['precision']:.2%}",
                f"{m.get('latency_avg', 0):.3f}", f"{m.get('latency_max', 0):.3f}",
            ])

        # ---- 相对 none 基线的提升 ----
        base = all_metrics["none"]
        delta_rows = []
        for name in RERANKERS:
            m = all_metrics[name]
            delta_rows.append([
                name,
                f"{(m['accuracy'] - base['accuracy']) * 100:+.1f} pp",
                f"{(m['recall'] - base['recall']) * 100:+.1f} pp",
                f"{(m['mrr'] - base['mrr']):+.4f}",
                f"{m.get('latency_avg', 0) - base.get('latency_avg', 0):+.3f}",
            ])

        table = md_table(
            ["重排器", f"准确率@{TOP_K}", f"召回率@{TOP_K}", f"HitRate@{TOP_K}",
             "MRR", f"Precision@{TOP_K}", "平均耗时(s)", "最大耗时(s)"],
            summary_rows)
        delta = md_table(["重排器", "准确率提升", "召回率提升", "MRR 提升", "耗时变化(s)"],
                         delta_rows)

        # 选优：准确率优先、同分看召回率、再看耗时
        best = max(RERANKERS, key=lambda n: (all_metrics[n]["accuracy"],
                                             all_metrics[n]["recall"],
                                             -all_metrics[n].get("latency_avg", 9e9)))
        best_m = all_metrics[best]
        evidence = (
            f"最优重排器：**{best}**，准确率 {best_m['accuracy']:.2%}"
            f"（目标 ≥{ACC_TARGET:.0%}，"
            f"{'达标' if best_m['accuracy'] >= ACC_TARGET else '未达标'}），"
            f"召回率 {best_m['recall']:.2%}"
            f"（目标 ≥{REC_RATIO_TARGET:.0%}，"
            f"{'达标' if best_m['recall'] >= REC_RATIO_TARGET else '未达标'}）。"
        )
        print("\n" + evidence)

        blocks = [
            "本报告由 `src/rerank_comparison.py` 自动生成，"
            "真实建索引、真实检索、真实统计。\n\n"
            f"- 检索策略：`{STRATEGY}` + `{FUSION}` 融合\n"
            f"- 召回 Top-{RECALL_K} -> 重排 -> 输出 Top-{TOP_K}\n"
            f"- 问题集：{len(WO06_QA_SET)} 问（兴图新科 10 + 力源信息 4）\n"
            f"- 重排器：{', '.join(RERANKERS)}\n"
            "- 指标：准确率=Top-5 覆盖全部关键信息点的问题占比；"
            "召回率=关键信息点被 Top-5 覆盖的平均比例；"
            "HitRate/MRR/Precision 为标准检索指标\n",
            "## 汇总对比\n\n" + table,
            "## 相对 none 基线的增益\n\n" + delta,
            "## 结论\n\n" + evidence +
            "\n\n- TF-IDF 重排零成本、无网络依赖，适合作为默认档或级联第一级；\n"
            "- LLM 重排精度高但引入网络延迟，建议只对粗排 Top-10 使用（级联）；\n"
            "- 自适应重排随用户反馈积累而持续变好，适合线上长期运行；\n"
            "- Cascade（TF-IDF 粗排 -> LLM 精排）在精度与 3 秒响应约束之间取平衡。",
        ]
        for name in RERANKERS:
            qstats, latencies = all_qstats[name]
            rows = per_question_rows(qstats, latencies)
            blocks.append(f"## 重排器 `{name}` 逐问题明细\n\n"
                          + md_table(["问题ID", "问题", "关键信息点覆盖", "可答对", "耗时(s)"],
                                     rows))
        # 最优配置的检索答案证据
        ev_lines = []
        for q in WO06_QA_SET:
            docs = details[best].get(str(q.qid), [])
            ev_lines.append(f"**问题{q.qid}**：{q.question}\n\n" +
                            "\n".join(passage_lines(
                                [{"text": d["snippet"], "doc": d["doc"], "page": d["page"],
                                  "type": d["type"], "score": d["score"]} for d in docs], k=3)))
        blocks.append(f"## 最优配置（{best}）检索答案证据（每问 Top-3 片段）\n\n"
                      + "\n\n".join(ev_lines))

        json_path = save_json(RESULTS_DIR / "rerank_comparison.json", {
            "工单编号": "人工智能NLP-RAG-混合检索任务",
            "strategy": STRATEGY, "fusion": FUSION,
            "recall_k": RECALL_K, "top_k": TOP_K,
            "rerankers": RERANKERS,
            "metrics": all_metrics,
            "best": best,
            "targets": {"accuracy": ACC_TARGET, "recall": REC_RATIO_TARGET},
            "details": details,
        })
        md_path = save_markdown(RESULTS_DIR / "rerank_comparison.md",
                                "工单06 重排算法对比实验报告", blocks)
        print(f"[完成] {json_path}\n[完成] {md_path}")
    finally:
        rerank_mod.FEEDBACK_FILE = original_feedback


if __name__ == "__main__":
    main()
