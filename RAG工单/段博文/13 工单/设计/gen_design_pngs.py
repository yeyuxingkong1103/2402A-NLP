# -*- coding: utf-8 -*-
"""工单13 设计PNG：5张架构图/流程图/对比图。"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent

def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# 01 基线检索流水线流程图
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
boxes = [
    (0.5, 4.5, 2.2, 1.0, "用户Query", "#3498db"),
    (0.5, 2.2, 2.2, 1.0, "查询编码\n0.06s", "#3498db"),
    (4.0, 5.5, 2.4, 1.0, "向量召回\n0.01s", "#2ecc71"),
    (4.0, 3.2, 2.4, 1.0, "BM25召回\n0.003s", "#1abc9c"),
    (7.5, 4.5, 2.2, 1.0, "RRF融合\n<1ms", "#f1c40f"),
    (10.5, 4.5, 3.0, 1.0, "CrossEncoder重排\n8.87s (99.2%)", "#e74c3c"),
    (10.5, 2.2, 3.0, 1.0, "分数过滤+组装\n<1ms", "#95a5a6"),
    (10.5, 0.3, 3.0, 0.9, "返回检索结果\n~9.0s", "#ecf0f1"),
]
for x,y,w,h,t,c in boxes:
    rect = plt.Rectangle((x,y), w, h, facecolor=c, edgecolor="black", linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
arrows = [
    (1.6,4.5,1.6,3.2), (1.6,3.5,4.0,4.0), (1.6,3.5,4.0,3.7),
    (5.2,5.0,7.5,5.0), (5.2,3.7,7.5,4.9),
    (8.6,5.0,10.5,5.0), (12.0,4.5,12.0,3.2), (12.0,2.2,12.0,1.2),
]
for x1,y1,x2,y2 in arrows:
    ax.annotate("", (x2,y2), (x1,y1), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.set_title("01 基线检索流水线流程图（瓶颈：CrossEncoder重排占99.2%）", fontsize=14, fontweight="bold")
fig.tight_layout(); save(fig, "01-基线检索流水线.png")

# 02 优化检索流水线
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
boxes2 = [
    (0.3, 4.2, 2.0, 1.0, "用户Query", "#3498db"),
    (0.3, 2.0, 2.0, 1.0, "查询编码\n0.06s", "#3498db"),
    (3.0, 5.5, 2.4, 1.0, "向量召回\n并行", "#2ecc71"),
    (3.0, 3.2, 2.4, 1.0, "BM25召回\n并行", "#1abc9c"),
    (6.0, 4.5, 2.0, 1.0, "RRF融合\n候选截断 16→6", "#f1c40f"),
    (8.5, 4.5, 2.6, 1.0, "CrossEncoder\n输入截断 500→256", "#e74c3c"),
    (8.5, 2.0, 2.6, 1.0, "过滤+组装", "#95a5a6"),
    (11.5, 4.5, 2.0, 1.0, "缓存命中\n0.001s", "#8e44ad"),
    (11.5, 0.3, 2.0, 0.9, "结果 <3s", "#ecf0f1"),
]
for x,y,w,h,t,c in boxes2:
    rect = plt.Rectangle((x,y), w, h, facecolor=c, edgecolor="black", linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
# 并行标注
ax.plot([2.3,2.3], [3.7,4.2], color="black", lw=1.5)
ax.annotate("", (3.0,6.0), (2.3,4.2), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.annotate("", (3.0,3.7), (2.3,3.7), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.plot([5.4,6.5], [6.0,5.0], color="black", lw=1.5); ax.plot([5.4,6.5], [3.7,5.0], color="black", lw=1.5)
ax.annotate("", (6.5,5.0), (6.5,5.0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
for x1,y1,x2,y2 in [(7.0,5.0,8.5,5.0),(9.8,4.5,9.8,3.0),(9.8,2.0,9.8,1.2),(12.5,4.5,12.5,1.2)]:
    ax.annotate("", (x2,y2), (x1,y1), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
# 缓存分支
ax.annotate("", (11.5,5.0), (9.8,5.0), arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5))
ax.text(10.5, 5.3, "命中", fontsize=10, color="#8e44ad")
ax.set_title("02 优化检索流水线（并行召回+候选截断+输入截断+缓存）", fontsize=14, fontweight="bold")
fig.tight_layout(); save(fig, "02-优化检索流水线.png")

# 03 瓶颈定位饼图
fig, ax = plt.subplots(figsize=(8, 8))
sizes = [8.868, 0.058, 0.010, 0.003]
labels = ["CrossEncoder重排 99.2%", "查询编码 0.7%", "向量召回 0.1%", "其他 <0.1%"]
colors3 = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]
explode = (0.06, 0, 0, 0)
ax.pie(sizes, explode=explode, labels=labels, colors=colors3, autopct="", startangle=90,
       textprops={"fontsize": 12})
ax.set_title("03 基线检索瓶颈定位（10条查询平均）", fontsize=14, fontweight="bold")
fig.tight_layout(); save(fig, "03-瓶颈定位饼图.png")

# 04 优化效果柱状图
fig, ax = plt.subplots(figsize=(9, 5.5))
cats = ["平均耗时", "P95耗时", "最大耗时"]
before = [8.94, 11.28, 12.07]
after  = [1.71, 1.84, 1.95]
x = range(len(cats))
ax.barh([i-0.18 for i in x], before, 0.35, label="优化前", color="#e74c3c")
ax.barh([i+0.18 for i in x], after, 0.35, label="优化后", color="#2ecc71")
for i,(vb,va) in enumerate(zip(before, after)):
    ax.text(vb+0.15, i-0.18, f"{vb:.2f}s", va="center", fontsize=11, color="#c0392b")
    ax.text(va+0.05, i+0.18, f"{va:.2f}s", va="center", fontsize=11, color="#27ae60", fontweight="bold")
ax.axvline(3.0, color="#c0392b", ls="--", lw=2)
ax.text(3.05, 2.3, "验收线 3S", color="#c0392b", fontsize=12)
ax.set_yticks(x); ax.set_yticklabels(cats)
ax.set_xlim(0, 14)
ax.set_xlabel("耗时（秒）", fontsize=12)
ax.set_title("04 优化前后耗时对比（CPU 环境，10 条法律查询）", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
fig.tight_layout(); save(fig, "04-优化前后耗时对比.png")

# 05 优化措施对应提速效果
fig, ax = plt.subplots(figsize=(11, 6))
measures = ["基线\n(~9.0s)", "O1\n单次编码", "O2\n并行召回", "O3\n候选截断\n16→6", "O4\n输入截断\n500→256", "O5\n查询缓存"]
times  = [8.94, 8.94, 8.86, 5.90, 1.71, 0.001]
bar_colors = ["#e74c3c", "#95a5a6", "#95a5a6", "#f39c12", "#f39c12", "#8e44ad"]
for i,(m,t,c) in enumerate(zip(measures, times, bar_colors)):
    ax.bar(i, t, color=c, edgecolor="black", linewidth=1.5)
    ax.text(i, t+0.2, f"{t:.2f}s" if t >= 0.01 else "<1ms",
            ha="center", fontsize=11, fontweight="bold")
ax.axhline(3.0, color="#c0392b", ls="--", lw=2)
ax.text(4.8, 3.2, "验收线 3S", color="#c0392b", fontsize=12)
ax.set_xticks(range(len(measures)))
ax.set_xticklabels(measures, fontsize=11)
ax.set_ylabel("耗时（秒）", fontsize=12)
ax.set_title("05 逐步优化效果：从基线到缓存命中", fontsize=14, fontweight="bold")
fig.tight_layout(); save(fig, "05-逐步优化效果.png")

print("全部 5 张 PNG 已生成")
