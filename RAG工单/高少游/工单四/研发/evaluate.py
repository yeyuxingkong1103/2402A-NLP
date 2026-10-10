# -*- coding: utf-8 -*-
"""检索与问答评估脚本（图像内容解析及检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

职责：
1. 逐题跑通 16 个验收问题，记录「答案文本 / 命中线索 / 端到端耗时 / 来源页码」；
2. 计算准确率（答案命中参考答案关键线索即判对）与平均响应时间；
3. 对照验收标准（准确率 ≥ 90%，响应 ≤ 3s）给出结论；
4. 输出 JSON 明细 + Markdown 报告 + 可视化图表（matplotlib）。

用法：
    python evaluate.py                 # 仅评估【优化后】链路
    python evaluate.py --baseline      # 追加【优化前】基线对照（需本地 LLM）
    python evaluate.py --ablation      # 追加消融（baseline_extractive / optimized_llm）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.evaluate_keys import KEY_TOKENS, hit, hit_detail
from src.qa_engine import QAEngine, retrieval_hit


def load_questions() -> list[dict]:
    return json.loads(Path(config.QUESTIONS_PATH).read_text(encoding="utf-8"))


def run_mode(engine: QAEngine, mode: str, questions: list[dict]) -> dict:
    """跑一个链路的全部题目，返回统计与逐题明细。"""
    rows = []
    ok = 0
    total_elapsed = 0.0
    for q in questions:
        qid = q["id"]
        tokens = KEY_TOKENS.get(qid, q.get("key_tokens", []))
        if mode == "optimized":
            res = engine.answer_optimized(q["question"])
        elif mode == "baseline":
            res = engine.answer_baseline(q["question"])
        elif mode == "baseline_extractive":
            res = engine.answer_baseline_extractive(q["question"])
        elif mode == "optimized_llm":
            res = engine.answer_optimized_llm(q["question"])
        else:
            raise ValueError(mode)

        detail = hit_detail(res.answer, tokens)
        passed = bool(detail["hit"])
        ok += int(passed)
        total_elapsed += res.elapsed
        rows.append({
            "id": qid,
            "type": q.get("type", ""),
            "question": q["question"],
            "answer": res.answer,
            "reference": q.get("reference", ""),
            "key_tokens": tokens,
            "matched": detail["matched"],
            "passed": passed,
            "elapsed": round(res.elapsed, 3),
            "sources": [{"page": s["page"], "source": s["source"],
                         "ctype": s.get("ctype", ""), "score": s.get("score", 0)}
                        for s in res.sources],
            "doc_source": getattr(res, "doc_source", ""),
            "retrieval_hit": retrieval_hit(res.sources, tokens) if res.sources else False,
        })
        flag = "✔" if passed else "’"
        print(f"  [{flag}] id={qid:<4} {res.elapsed:5.2f}s  {res.answer[:70]}")

    n = len(questions)
    return {
        "mode": mode,
        "n": n,
        "correct": ok,
        "accuracy": round(ok / n, 4) if n else 0.0,
        "avg_elapsed": round(total_elapsed / n, 3) if n else 0.0,
        "max_elapsed": round(max((r["elapsed"] for r in rows), default=0.0), 3),
        "rows": rows,
    }


def _plot(summary: dict, out_dir: Path) -> None:
    """生成准确率与响应时间图表。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # 图表失败不影响评估
        print("  (matplotlib 不可用，跳过图表):", exc)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = summary["rows"]
    # 1) 逐题耗时
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    ids = [str(r["id"]) for r in rows]
    el = [r["elapsed"] for r in rows]
    colors = ["#2e7d32" if e <= config.TARGET_RESPONSE_SECONDS else "#c62828" for e in el]
    ax.bar(ids, el, color=colors)
    ax.axhline(config.TARGET_RESPONSE_SECONDS, color="#1565c0", ls="--", lw=1.2,
               label=f"目标 {config.TARGET_RESPONSE_SECONDS:.0f}s")
    ax.set_title("逐题端到端响应时间（绿色=达标）")
    ax.set_xlabel("问题 ID")
    ax.set_ylabel("秒")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "response_time.png")
    plt.close(fig)

    # 2) 准确率环形图
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=140)
    acc = summary["accuracy"]
    ax.pie([acc, 1 - acc], labels=["命中", "未命中"], autopct="%1.1f%%",
           colors=["#2e7d32", "#e0e0e0"], startangle=90,
           wedgeprops=dict(width=0.38, edgecolor="w"))
    ax.set_title(f"答案准确率（{summary['correct']}/{summary['n']}）")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png")
    plt.close(fig)

    # 3) 按题型准确率
    by_type: dict[str, list[int]] = {}
    for r in rows:
        by_type.setdefault(r["type"] or "其他", []).append(int(r["passed"]))
    if by_type:
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=140)
        names = list(by_type)
        vals = [sum(v) / len(v) * 100 for v in by_type.values()]
        ax.bar(names, vals, color=["#1565c0", "#ef6c00", "#6a1b9a", "#00838f"][:len(names)])
        ax.set_ylim(0, 105)
        ax.set_ylabel("准确率 (%)")
        ax.set_title("分题型准确率")
        for i, v in enumerate(vals):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_by_type.png")
        plt.close(fig)


def _md_report(summary: dict, out_path: Path) -> None:
    """生成 Markdown 评测报告。"""
    rows = summary["rows"]
    acc = summary["accuracy"]
    pass_acc = acc >= config.TARGET_ACCURACY
    pass_time = summary["max_elapsed"] <= config.TARGET_RESPONSE_SECONDS
    lines = [
        "# 工单四 · 图像内容解析及检索优化 —— 问答评测报告",
        "",
        "工单编号：人工智能 NLP-RAG-图像内容解析及检索优化",
        "",
        "## 一、评测结论",
        "",
        f"- 题目总数：**{summary['n']}**，答对：**{summary['correct']}**",
        f"- 答案准确率：**{acc * 100:.2f}%**（验收标准 ≥ 90%）→ {'**达标**' if pass_acc else '未达标'}",
        f"- 平均响应时间：**{summary['avg_elapsed']:.2f}s**，最大 **{summary['max_elapsed']:.2f}s**"
        f"（验收标准 ≤ 3s）→ {'**达标**' if pass_time else '未达标'}",
        "",
        "## 二、逐题明细",
        "",
        "| ID | 题型 | 耗时(s) | 是否命中 | 命中线索 | 答案摘要 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ans = r["answer"].replace("\n", " ").replace("|", "/")
        lines.append(
            f"| {r['id']} | {r['type']} | {r['elapsed']:.2f} | "
            f"{'✔' if r['passed'] else '✘'} | {', '.join(r['matched']) or '—'} | {ans[:60]} |"
        )
    lines += ["", "## 三、逐题完整答案", ""]
    for r in rows:
        lines += [
            f"### id {r['id']}（{r['type']}）",
            "",
            f"- 问题：{r['question']}",
            f"- 参考答案：{r['reference']}",
            f"- 系统答案：{r['answer']}",
            f"- 命中线索：{', '.join(r['matched']) or '无'}",
            f"- 来源页码：{', '.join(str(s['page']) for s in r['sources'][:5])}",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="追加基线（LLM）对照")
    ap.add_argument("--ablation", action="store_true", help="追加消融对照")
    args = ap.parse_args()

    questions = load_questions()
    engine = QAEngine()
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 预热：加载向量库 / BM25 / Chinese-CLIP 权重，消除首问的模型冷启动耗时。
    # 该耗时（十余秒）为进程级一次性开销，不计入单题响应时间。
    print("预热模型（向量库 / BM25 / Chinese-CLIP）...")
    t_warm = time.time()
    engine.warmup()
    print(f"  预热完成，用时 {time.time() - t_warm:.2f}s")

    print("=" * 68)
    print(f"【优化后链路】评估 {len(questions)} 题")
    print("=" * 68)
    summary = run_mode(engine, "optimized", questions)

    all_summaries = {"optimized": summary}

    if args.baseline:
        print("=" * 68)
        print("【优化前基线链路（LLM）】评估")
        print("=" * 68)
        all_summaries["baseline"] = run_mode(engine, "baseline", questions)

    if args.ablation:
        for mode in ("baseline_extractive", "optimized_llm"):
            print("=" * 68)
            print(f"【消融：{mode}】评估")
            print("=" * 68)
            all_summaries[mode] = run_mode(engine, mode, questions)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_accuracy": config.TARGET_ACCURACY,
        "target_seconds": config.TARGET_RESPONSE_SECONDS,
        "summaries": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      for k, v in all_summaries.items()},
        "details": {k: v["rows"] for k, v in all_summaries.items()},
    }
    (out_dir / "eval_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot(summary, Path(config.FIGURE_OUT_DIR))
    _md_report(summary, out_dir / "评测报告.md")

    print("-" * 68)
    print(f"准确率: {summary['accuracy'] * 100:.2f}%   "
          f"平均耗时: {summary['avg_elapsed']:.2f}s   最大: {summary['max_elapsed']:.2f}s")
    print(f"结果: {out_dir / 'eval_results.json'}")
    print(f"报告: {out_dir / '评测报告.md'}")
    print(f"图表: {config.FIGURE_OUT_DIR}")


if __name__ == "__main__":
    main()