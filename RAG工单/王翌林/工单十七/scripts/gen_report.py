# -*- coding: utf-8 -*-
"""读取基线/优化版压测结果，生成对比图（延迟/吞吐/内存曲线）与报告。"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "/home/dabaie/.fonts/simhei.ttf"
if os.path.isfile(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/home/dabaie/code/工单/工单十七/docs/screenshots"
LT = "/home/dabaie/code/工单/工单十七/loadtest"
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(LT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def main():
    base = load("result_base.json"); opt = load("result_opt.json")
    if not base or not opt:
        print("结果文件缺失"); return
    labels = ["基线", "优化后"]
    # 1. 延迟对比
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(2)
    for i, (m, c) in enumerate(zip(["avg_ms", "p95_ms", "p99_ms"], ["#888", "#d62728", "#ff7f0e"])):
        ax.bar([xi + i * 0.25 for xi in x], [base[m], opt[m]], 0.25, label=m, color=c)
    ax.set_xticks([xi + 0.25 for xi in x]); ax.set_xticklabels(labels)
    ax.set_ylabel("延迟 (ms)"); ax.set_title("响应延迟对比（越低越好）")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/01_latency_compare.png", dpi=130); plt.close(fig)

    # 2. 吞吐对比
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [base["throughput_rps"], opt["throughput_rps"]], color=["#d62728", "#2ca02c"])
    ax.set_ylabel("吞吐 (req/s)"); ax.set_title("吞吐量对比（越高越好）")
    for i, v in enumerate([base["throughput_rps"], opt["throughput_rps"]]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/02_throughput_compare.png", dpi=130); plt.close(fig)

    # 3. 内存增长曲线
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(base["mem_samples_mb"], label=f"基线 (增长 {base['rss_growth_pct']}%)", color="#d62728")
    ax.plot(opt["mem_samples_mb"], label=f"优化后 (增长 {opt['rss_growth_pct']}%)", color="#2ca02c")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("RSS 内存 (MB)")
    ax.set_title("压测期间内存占用曲线（基线持续上涨=泄漏，优化后平稳）")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/03_memory_curve.png", dpi=130); plt.close(fig)

    # 4. 指标汇总表（文本）
    rows = [
        ("平均延迟(ms)", base["avg_ms"], opt["avg_ms"]),
        ("P95延迟(ms)", base["p95_ms"], opt["p95_ms"]),
        ("P99延迟(ms)", base["p99_ms"], opt["p99_ms"]),
        ("吞吐(req/s)", base["throughput_rps"], opt["throughput_rps"]),
        ("错误数", base["errors"], opt["errors"]),
        ("内存增长(%)", base["rss_growth_pct"], opt["rss_growth_pct"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axis("off")
    tbl = ax.table(cellText=[[r[0], r[1], r[2]] for r in rows],
                   colLabels=["指标", "基线", "优化后"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
    ax.set_title("优化前后关键指标汇总", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/04_metrics_table.png", dpi=130); plt.close(fig)
    print("charts saved to", OUT)


if __name__ == "__main__":
    main()
