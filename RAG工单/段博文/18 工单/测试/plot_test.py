# -*- coding: utf-8 -*-
"""基于全量质检报告生成测试统计图，并同步 HTML 简报到测试目录。"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
REPORT = BASE / "研发" / "src" / "results" / "quality_report.json"
TEST = BASE / "测试"

rep = json.loads(REPORT.read_text(encoding="utf-8"))
C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GRAY = (
    "#4a90d9", "#5cb85c", "#f0ad4e", "#d9534f", "#9b59b6", "#7f8c8d")


# ---------- 图1：PDF 类型分流 ----------
def fig1():
    types = rep["summary"]["pdf_type_counts"]
    labels = {"Scan_PDF": "扫描型", "Hybrid_PDF": "混合型", "Text_PDF": "文字型"}
    vals = [labels.get(k, k) for k in types]
    counts = list(types.values())
    colors = [{"Scan_PDF": C_RED, "Hybrid_PDF": C_ORANGE, "Text_PDF": C_GREEN}.get(k) for k in types]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(vals, counts, color=colors)
    for i, c in enumerate(counts):
        axes[0].text(i, c + 8, str(c), ha="center", fontsize=11)
    axes[0].set_title("PDF 页面类型分流（共1700份）")
    axes[0].set_ylabel("文件数")
    axes[1].pie(counts, labels=vals, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[1].set_title("类型占比")
    fig.tight_layout()
    fig.savefig(TEST / "01-PDF类型分流.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("01-PDF类型分流.png")


# ---------- 图2：长度分布 ----------
def fig2():
    L = rep["length"]
    bins = L["bins_distribution"]
    names = [b["range"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bars = axes[0].bar(names, counts, color=[C_GRAY, C_GREEN, C_BLUE, C_ORANGE, C_RED, C_PURPLE])
    for b, c in zip(bars, counts):
        if c:
            axes[0].text(b.get_x() + b.get_width() / 2, c + 6, str(c), ha="center", fontsize=9)
    axes[0].set_title("文档长度区间分布（字符）")
    axes[0].tick_params(axis="x", labelsize=8)

    pcts = L["percentiles"]
    axes[1].plot(list(pcts.keys()), list(pcts.values()), "o-", color=C_BLUE)
    for k, v in pcts.items():
        axes[1].text(k, v + 150, f"{v:,}", ha="center", fontsize=8)
    axes[1].set_title("长度分位数曲线")
    axes[1].set_ylabel("字符数")
    fig.tight_layout()
    fig.savefig(TEST / "02-文档长度分布.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("02-文档长度分布.png")


# ---------- 图3：解析路由 ----------
def fig3():
    routes = rep["summary"]["routing_counts"]
    names_cn = {
        "OCRParser": "OCR解析", "HybridParser": "混合解析", "DirectParser": "直接解析",
        "VersionReviewNode": "版本审核", "SecurityReviewNode": "安全审核", "ReviewNode": "损坏审核"}
    names = [names_cn.get(k, k) for k in routes]
    counts = list(routes.values())
    colors = plt.cm.Set2(range(len(names)))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(names, counts, color=colors)
    for b, c in zip(bars, counts):
        ax.text(c + 8, b.get_y() + b.get_height() / 2, str(c), va="center", fontsize=10)
    ax.set_title("工作流解析路由分布（质检决策结果）")
    ax.set_xlabel("文档数")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(TEST / "03-解析路由分布.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("03-解析路由分布.png")


# ---------- 图4：版本冲突 + 敏感信息 ----------
def fig4():
    pairs = rep["duplicates"]["simhash_pairs"]
    dist_counter = {}
    for p in pairs:
        dist_counter[p["hamming_distance"]] = dist_counter.get(p["hamming_distance"], 0) + 1
    sens = rep["summary"]["sensitive_by_type"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    if dist_counter:
        ds = sorted(dist_counter)
        axes[0].bar([f"距离{d}" for d in ds], [dist_counter[d] for d in ds], color=C_PURPLE)
        for i, d in enumerate(ds):
            axes[0].text(i, dist_counter[d] + 0.2, str(dist_counter[d]), ha="center")
    axes[0].set_title("SimHash 待确认版本冲突（共%d对）" % len(pairs))

    if sens:
        axes[1].bar(list(sens.keys()), list(sens.values()), color=C_RED)
        for i, (k, v) in enumerate(sens.items()):
            axes[1].text(i, v + 0.05, str(v), ha="center")
    axes[1].set_title("敏感信息待审核（共%d条）" % len(rep["sensitive_review"]))
    fig.tight_layout()
    fig.savefig(TEST / "04-版本冲突与敏感信息.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("04-版本冲突与敏感信息.png")


fig1(); fig2(); fig3(); fig4()

# 同步 HTML 简报
shutil.copy(BASE / "研发" / "src" / "results" / "quality_report.html",
            TEST / "文档质检报告.html")
print("文档质检报告.html 已同步")
