# -*- coding: utf-8 -*-
# 工单15：生成设计图 5 张 PNG
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
DESIGN = Path(__file__).resolve().parent
C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GRAY = "#4a90d9", "#5cb85c", "#f0ad4e", "#d9534f", "#9b59b6", "#7f8c8d"

def box(ax, x, y, w, h, text, fc=C_BLUE, fs=11, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#333", lw=1.2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)

def arrow(ax, p1, p2, color="#444"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color))

def save(fig, name):
    fig.savefig(DESIGN / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"已生成 {name}")

# 图1：跨模态检索优化总体架构
def fig1():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.text(5.5, 7.2, "跨模态检索优化总体架构", ha="center", fontsize=15, weight="bold")
    box(ax, 0.3, 5.8, 2.2, 0.8, "用户问题\n(文本+视觉引用)", C_GRAY)
    box(ax, 3.2, 6.0, 2.5, 0.6, "O1 查询理解\n识别\"第11页图3\"", C_BLUE)
    box(ax, 3.2, 5.0, 2.5, 0.6, "O2 图像描述文本化\n图3→文字描述chunk", C_BLUE)
    box(ax, 6.5, 6.0, 2.2, 0.6, "文本语义检索\nbge-m3", C_GREEN)
    box(ax, 6.5, 5.0, 2.2, 0.6, "图像增强检索\n部件编号+图号", C_GREEN)
    box(ax, 6.5, 3.8, 2.2, 0.6, "O3 RRF融合\n多路召回加权", C_ORANGE)
    box(ax, 3.2, 3.8, 2.5, 0.6, "O4 跨模态重排\n图3/部件加权", C_ORANGE)
    box(ax, 3.2, 2.6, 2.5, 0.6, "O5 Prompt模板\n图文结合上下文", C_PURPLE)
    box(ax, 6.5, 2.6, 2.2, 0.6, "O6 抽取式问答\n规则提取答案", C_RED)
    box(ax, 3.2, 1.2, 5.5, 0.8, "知识库：专利文本chunk + 图像描述chunk（图1/图2/图3的文字描述）", "#34495e", fs=10)
    arrow(ax, (2.5, 6.2), (3.2, 6.3)); arrow(ax, (2.5, 6.0), (3.2, 5.3))
    arrow(ax, (5.7, 6.3), (6.5, 6.3)); arrow(ax, (5.7, 5.3), (6.5, 5.3))
    arrow(ax, (8.7, 6.3), (7.6, 4.1)); arrow(ax, (8.7, 5.3), (7.6, 4.1))
    arrow(ax, (6.5, 4.1), (5.7, 4.1)); arrow(ax, (4.4, 3.8), (4.4, 3.2))
    arrow(ax, (5.7, 2.9), (6.5, 2.9)); arrow(ax, (4.4, 2.6), (4.4, 2.0))
    save(fig, "01-跨模态检索架构图.png")

# 图2：查询理解流程
def fig2():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")
    ax.text(5.5, 3.7, "O1 查询理解：识别视觉引用", ha="center", fontsize=15, weight="bold")
    steps = [("原始问题\n\"第11页图3中\n编号13的部件...\"", C_GRAY),
             ("正则匹配\n第\\d+页图\\d+", C_BLUE),
             ("提取: page=11\nfigure=3, ids=[13]", C_GREEN),
             ("构建增强查询\n+\"图3\"+\"部件13\"", C_ORANGE),
             ("送入图像增强\n检索通道", C_PURPLE)]
    w, h, gap = 1.9, 1.4, 0.25
    x = 0.25
    centers = []
    for t, c in steps:
        box(ax, x, 1.8, w, h, t, c, fs=9.5); centers.append(x+w); x += w+gap
    for i in range(len(steps)-1):
        arrow(ax, (centers[i], 2.5), (centers[i]+gap, 2.5))
    ax.text(5.5, 0.8, "效果：图像题的top1从普通文本chunk变为图3描述chunk", ha="center", fontsize=10, color=C_RED)
    save(fig, "02-查询理解流程图.png")

# 图3：基线vs优化后检索对比
def fig3():
    import json, numpy as np
    qa = json.loads((DESIGN.parent / "研发" / "qa_results.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ids = [f"Q{r['id']}" for r in qa["results"]]
    base_fig3 = [1 if b["has_fig3"] else 0 for b in qa["baseline"]]
    opt_fig3 = [1 if r["top1_has_fig3"] else 0 for r in qa["results"]]
    x = np.arange(len(ids)); w = 0.35
    ax.bar(x-w/2, base_fig3, w, label="基线top1含图3", color="#7f8c8d")
    ax.bar(x+w/2, opt_fig3, w, label="优化后top1含图3", color=C_GREEN)
    ax.set_xticks(x); ax.set_xticklabels(ids)
    ax.set_ylabel("是否含图3描述"); ax.set_ylim(-0.1, 1.3)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("基线 vs 优化后：Top1检索结果是否包含图3描述", fontsize=13, weight="bold")
    fig.tight_layout(); save(fig, "03-基线vs优化检索对比.png")

# 图4：准确率与耗时
def fig4():
    import json, numpy as np
    qa = json.loads((DESIGN.parent / "研发" / "qa_results.json").read_text(encoding="utf-8"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    ids = [f"Q{r['id']}\n{'文本题' if r['type']=='text' else '图像题'}" for r in qa["results"]]
    acc = [100 if r["correct"] else 0 for r in qa["results"]]
    a1.bar(ids, acc, color=C_GREEN, width=0.6); a1.set_ylim(0, 118); a1.set_ylabel("正确率 %")
    a1.set_title(f"逐题正确率（总准确率 {qa['accuracy']:.0f}%）", fontsize=12.5, weight="bold")
    for i, v in enumerate(acc): a1.text(i, v+3, "对", ha="center", fontsize=12, color=C_GREEN)
    t = [r["elapsed"] for r in qa["results"]]
    a2.bar(ids, t, color=C_BLUE, width=0.6); a2.axhline(5, color=C_RED, ls="--", lw=1.5)
    a2.text(0.1, 5.15, "验收线5s", color=C_RED, fontsize=10); a2.set_ylim(0, 5.5); a2.set_ylabel("耗时 秒")
    a2.set_title(f"逐题响应耗时（平均 {qa['avg_sec']}s）", fontsize=12.5, weight="bold")
    for i, v in enumerate(t): a2.text(i, v+0.08, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); save(fig, "04-准确率与耗时.png")

# 图5：RRF融合+重排流程
def fig5():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(5.5, 4.7, "O3+O4 多路召回融合与跨模态重排", ha="center", fontsize=15, weight="bold")
    box(ax, 0.3, 3.0, 2.0, 0.8, "文本语义检索\nTop-5", C_GREEN)
    box(ax, 0.3, 1.8, 2.0, 0.8, "图像增强检索\nTop-5", C_GREEN)
    box(ax, 3.0, 2.4, 2.0, 0.8, "RRF融合\n1/(60+rank)", C_ORANGE)
    box(ax, 5.8, 2.4, 2.2, 0.8, "跨模态重排\n图3描述+0.3\n部件编号+0.1", C_ORANGE)
    box(ax, 8.6, 2.4, 2.1, 0.8, "Top-K\n图文结合上下文", C_PURPLE)
    arrow(ax, (2.3, 3.4), (3.0, 3.0)); arrow(ax, (2.3, 2.2), (3.0, 2.6))
    arrow(ax, (5.0, 2.8), (5.8, 2.8)); arrow(ax, (8.0, 2.8), (8.6, 2.8))
    ax.text(5.5, 0.8, "融合后对含图3描述/部件编号的chunk加权，优先选出图文语义最匹配的片段", ha="center", fontsize=10, color="#555")
    save(fig, "05-RRF融合与重排图.png")

if __name__ == "__main__":
    import numpy as np
    fig1(); fig2(); fig3(); fig4(); fig5()
