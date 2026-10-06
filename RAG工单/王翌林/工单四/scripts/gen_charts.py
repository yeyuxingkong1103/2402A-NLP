# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
scripts/gen_charts.py —— 工单二对比评估图表生成（docs/screenshots/）

产出（matplotlib Agg，中文字体自动探测 Noto Sans CJK / WenQuanYi）：
  01_优化前.png   基线链路 10 题答案与延迟卡片
  02_优化后.png   优化链路 10 题答案与延迟卡片
  03_对比表.png   三链路六指标对比表
  04_响应时间.png 三链路逐题响应时间对比柱状图
  05_准确率对比.png 准确率/上下文召回提升柱状图
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT_DIR = "docs/screenshots"
COMPARE_JSON = "data/optimized/compare_eval.json"

# ---------- 工单二中文字体探测（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def _cjk_font():
    prefer = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
              "WenQuanYi Zen Hei", "AR PL UMing CN"]
    names = {f.name for f in font_manager.fontManager.ttflist}
    for p in prefer:
        if p in names:
            return p
    return None


plt.rcParams["font.family"] = [_cjk_font() or "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
C_BASE, C_OPT, C_LLM = "#8c8c8c", "#2f7df6", "#b8b8b8"


def _ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def card_answers(chain_rows: dict, title: str, fname: str, color: str):
    """链路答案卡片图：每题一行（问题 / 答案摘要 / 延迟 / 指标）（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    rows = chain_rows["rows"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(13, 1.05 * n + 1.2))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_title(f"{title}\n准确率 {chain_rows['summary']['accuracy']:.0%} | "
                 f"平均延迟 {chain_rows['summary']['avg_latency_ms']:.0f} ms",
                 fontsize=14, fontweight="bold", color="#1a1a2e", pad=12)
    for i, r in enumerate(rows):
        y = n - i - 0.5
        bg = "#f0f6ff" if r["accuracy"] else "#fff2f0"
        ax.add_patch(plt.Rectangle((0.01, y - 0.44), 0.98, 0.9, transform=ax.transAxes,
                                   facecolor=bg, edgecolor="#d0d7e2", lw=0.8, zorder=1))
        acc_tag = "✓ 正确" if r["accuracy"] else "✗ 未命中"
        ax.text(0.02, y + 0.26, f"{r['id']} [{r['type']}]  {r['question'][:38]}",
                transform=ax.transAxes, fontsize=9.5, fontweight="bold", zorder=2)
        ax.text(0.02, y - 0.18, f"{r['answer'][:70].replace(chr(10), ' ')}",
                transform=ax.transAxes, fontsize=8.5, color="#333", zorder=2)
        ax.text(0.985, y + 0.26, f"{r['latency_ms']:.0f} ms  {acc_tag}  F={r['faithfulness']:.1f}",
                transform=ax.transAxes, fontsize=9, ha="right", color=color,
                fontweight="bold", zorder=2)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"✅ {fname}")


def chart_compare_table(result: dict):
    """三链路六指标对比表（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    metrics = [("accuracy", "准确率", lambda v: f"{v:.0%}"),
               ("avg_latency_ms", "平均响应(ms)", lambda v: f"{v:.0f}"),
               ("faithfulness", "忠实度", lambda v: f"{v:.2f}"),
               ("relevance", "答案相关性", lambda v: f"{v:.2f}"),
               ("ctx_precision", "上下文精度", lambda v: f"{v:.2f}"),
               ("ctx_recall", "上下文召回", lambda v: f"{v:.0%}")]
    chains = [("baseline", "优化前(基线)"), ("optimized", "优化后(工单二)"), ("pure_llm", "纯LLM")]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    tb = ax.table(cellText=[[fmt(result["chains"][c]["summary"][m]) for m, _, fmt in metrics]
                            for c, label in chains],
                  rowLabels=[label for _, label in chains],
                  colLabels=[label for _, label, _ in metrics],
                  cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 2.0)
    for (r, c), cell in tb.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#e8eefc")
            cell.set_text_props(fontweight="bold")
        elif r == 2:  # 优化后行高亮
            cell.set_facecolor("#eaf3ff")
    ax.set_title("工单二 优化前后六指标对比（人工智能NLP-RAG-基于PDF文档的问答系统优化）",
                 fontsize=13, fontweight="bold", pad=16)
    fig.savefig(os.path.join(OUT_DIR, "03_对比表.png"), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("✅ 03_对比表.png")


def chart_latency(result: dict):
    """逐题响应时间对比柱状图（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    import numpy as np
    ids = [r["id"] for r in result["chains"]["optimized"]["rows"]]
    vals = {c: [r["latency_ms"] for r in result["chains"][c]["rows"]]
            for c in ("baseline", "optimized", "pure_llm")}
    x = np.arange(len(ids))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, vals["baseline"], w, label="优化前(基线)", color=C_BASE)
    ax.bar(x, vals["optimized"], w, label="优化后(工单二)", color=C_OPT)
    ax.bar(x + w, vals["pure_llm"], w, label="纯LLM", color=C_LLM)
    ax.axhline(3000, color="#e05252", ls="--", lw=1.2, label="目标 3000ms")
    ax.set_xticks(x, ids)
    ax.set_ylabel("响应时间 (ms)")
    ax.set_title("三链路逐题响应时间对比（人工智能NLP-RAG-基于PDF文档的问答系统优化）",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, "04_响应时间.png"), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("✅ 04_响应时间.png")


def chart_accuracy(result: dict):
    """准确率与上下文召回提升图（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    import numpy as np
    metrics = [("accuracy", "准确率"), ("ctx_recall", "上下文召回"),
               ("faithfulness", "忠实度"), ("relevance", "答案相关性"),
               ("ctx_precision", "上下文精度")]
    b = [result["chains"]["baseline"]["summary"][m] for m, _ in metrics]
    o = [result["chains"]["optimized"]["summary"][m] for m, _ in metrics]
    x = np.arange(len(metrics))
    w = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.bar(x - w / 2, b, w, label="优化前(基线)", color=C_BASE)
    ax.bar(x + w / 2, o, w, label="优化后(工单二)", color=C_OPT)
    for xi, (bv, ov) in enumerate(zip(b, o)):
        ax.text(xi - w / 2, bv + 0.02, f"{bv:.0%}" if bv <= 1 else f"{bv:.0f}",
                ha="center", fontsize=10)
        ax.text(xi + w / 2, ov + 0.02, f"{ov:.0%}" if ov <= 1 else f"{ov:.0f}",
                ha="center", fontsize=10, fontweight="bold", color=C_OPT)
    ax.set_xticks(x, [n for _, n in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_title("优化前后质量指标对比（人工智能NLP-RAG-基于PDF文档的问答系统优化）",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, "05_准确率对比.png"), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("✅ 05_准确率对比.png")


def gen_all(result: dict):
    """生成全部对比图表（工单：人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    _ensure_dir()
    card_answers(result["chains"]["baseline"], "优化前（工单一基线链路）答案明细",
                 "01_优化前.png", C_BASE)
    card_answers(result["chains"]["optimized"], "优化后（工单二混合检索+重排+父子块）答案明细",
                 "02_优化后.png", C_OPT)
    chart_compare_table(result)
    chart_latency(result)
    chart_accuracy(result)


if __name__ == "__main__":
    gen_all(json.load(open(COMPARE_JSON, encoding="utf-8")))
