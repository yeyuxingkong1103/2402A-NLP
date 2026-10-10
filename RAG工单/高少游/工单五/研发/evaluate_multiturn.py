# -*- coding: utf-8 -*-
"""多轮对话评测：准确率与响应时间。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

评测口径：
    - 单轮命中：答案文本覆盖该轮全部关键 token（key_tokens）判定为正确；
    - 准确率  ：命中轮数 / 总轮数（工单要求 ≥ 90%）；
    - 响应时间：每轮从提问到返回答案的墙钟耗时（工单要求 ≤ 3s）；
    - 输出    ：output/eval_results.json + output/评测报告.md + output/figures/*.png
"""
from __future__ import annotations

import json
import sys
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.conversation import Conversation
from src.qa_engine import MultiTurnQA

# 中文字体（Windows 自带黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _hit(answer: str, key_tokens: list[str]) -> bool:
    if not key_tokens:
        return bool(answer)
    return all(t in answer for t in key_tokens)


def main() -> int:
    script = json.loads(config.MULTITURN_PATH.read_text(encoding="utf-8"))
    turns = script["turns"]

    qa = MultiTurnQA()
    qa.warmup()          # 预热，剔除冷启动对首轮耗时的影响
    conv = Conversation()
    rows = []
    for item in turns:
        res = qa.ask(item["question"], conv)
        ok = _hit(res.answer, item.get("key_tokens", []))
        rows.append({
            "turn": item["turn"],
            "question": item["question"],
            "rewritten": res.rewritten,
            "answer": res.answer,
            "doc": res.doc,
            "answer_type": res.answer_type,
            "latency": res.latency,
            "hit": ok,
            "key_tokens": item.get("key_tokens", []),
            "reference": item.get("reference", ""),
        })

    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    acc = hits / n if n else 0.0
    lat = [r["latency"] for r in rows]
    avg_lat = mean(lat) if lat else 0.0
    max_lat = max(lat) if lat else 0.0

    summary = {
        "workorder": "人工智能 NLP-RAG-Query 理解优化任务",
        "turns": n,
        "hits": hits,
        "accuracy": round(acc, 4),
        "accuracy_target": config.TARGET_ACCURACY,
        "accuracy_pass": acc >= config.TARGET_ACCURACY,
        "avg_latency": round(avg_lat, 3),
        "max_latency": round(max_lat, 3),
        "latency_target": config.TARGET_RESPONSE_SECONDS,
        "latency_pass": max_lat <= config.TARGET_RESPONSE_SECONDS,
        "rows": rows,
    }
    (config.OUTPUT_DIR / "eval_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_report(summary)
    _plots(rows, summary)

    print(f"准确率 {hits}/{n} = {acc:.1%}（目标 ≥ {config.TARGET_ACCURACY:.0%}）"
          f" -> {'通过' if summary['accuracy_pass'] else '未通过'}")
    print(f"平均耗时 {avg_lat:.3f}s / 最大 {max_lat:.3f}s（目标 ≤ {config.TARGET_RESPONSE_SECONDS}s）"
          f" -> {'通过' if summary['latency_pass'] else '未通过'}")
    return 0


def _write_report(s: dict) -> None:
    lines = [
        "# 多轮问答评测报告",
        "",
        "> 工单编号：人工智能 NLP-RAG-Query 理解优化任务",
        "",
        "## 一、总体结论",
        "",
        f"- 评测轮数：**{s['turns']}**",
        f"- 命中轮数：**{s['hits']}**",
        f"- 准确率：**{s['accuracy']:.1%}**（目标 ≥ {s['accuracy_target']:.0%}）"
        f" → {'✅ 通过' if s['accuracy_pass'] else '❌ 未通过'}",
        f"- 平均响应：**{s['avg_latency']}s**，最大 **{s['max_latency']}s**"
        f"（目标 ≤ {s['latency_target']}s） → {'✅ 通过' if s['latency_pass'] else '❌ 未通过'}",
        "",
        "## 二、逐轮明细",
        "",
        "| 轮次 | 用户问题 | 改写问句 | 答案 | 文档 | 命中 | 耗时(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in s["rows"]:
        q = r["question"].replace("|", "｜")
        rw = (r["rewritten"] or "").replace("|", "｜")
        a = r["answer"].replace("|", "｜")
        lines.append(f"| {r['turn']} | {q} | {rw} | {a} | {r['doc'] or '-'} | "
                     f"{'✅' if r['hit'] else '❌'} | {r['latency']} |")
    lines += [
        "",
        "## 三、指标图",
        "",
        "![逐轮耗时](figures/latency.png)",
        "",
        "![准确率](figures/accuracy.png)",
    ]
    (config.OUTPUT_DIR / "评测报告.md").write_text("\n".join(lines), encoding="utf-8")


def _plots(rows: list[dict], s: dict) -> None:
    figdir = config.FIGURE_OUT_DIR
    figdir.mkdir(parents=True, exist_ok=True)

    # 逐轮耗时
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [f"T{r['turn']}" for r in rows]
    ys = [r["latency"] for r in rows]
    colors = ["#2e7d32" if y <= config.TARGET_RESPONSE_SECONDS else "#c62828" for y in ys]
    ax.bar(xs, ys, color=colors)
    ax.axhline(config.TARGET_RESPONSE_SECONDS, color="#c62828", ls="--", lw=1.2,
               label=f"目标 ≤ {config.TARGET_RESPONSE_SECONDS}s")
    for i, y in enumerate(ys):
        ax.text(i, y + 0.02, f"{y:.2f}", ha="center", fontsize=9)
    ax.set_title("多轮问答逐轮响应时间")
    ax.set_ylabel("秒")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "latency.png", dpi=150)
    plt.close(fig)

    # 准确率
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["准确率"], [s["accuracy"] * 100], color="#1565c0", width=0.5)
    ax.axhline(s["accuracy_target"] * 100, color="#c62828", ls="--",
               label=f"目标 ≥ {s['accuracy_target']:.0%}")
    ax.set_ylim(0, 110)
    ax.text(0, s["accuracy"] * 100 + 3, f"{s['accuracy']:.1%}", ha="center")
    ax.set_title("多轮问答准确率")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "accuracy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())