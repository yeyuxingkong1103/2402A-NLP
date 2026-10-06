# -*- coding: utf-8 -*-
"""可视化图表生成脚本
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

读取 output/evaluation_report.json，生成优化前后对比图表与系统流程图：
    01_metrics_before_after.png   核心指标对比
    02_per_question_hit.png       逐题命中对比
    03_latency_compare.png        逐题响应时间对比
    04_pipeline_flow.png          优化后 RAG 链路流程图
    05_chunking_compare.png       分块策略对比示意
    06_optimization_layers.png    四层优化点总览

用法：python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_BASE = "#f97316"     # 优化前
C_OPT = "#16a34a"      # 优化后
C_TEXT = "#1f2937"
C_GRID = "#e5e7eb"


def _fig_dir() -> Path:
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return config.FIGURE_DIR


def _save(fig, name: str, tries: int = 5) -> None:
    """保存图片（受限环境下偶发 PermissionError，重试若干次）。"""
    import time

    path = _fig_dir() / name
    for _ in range(tries):
        try:
            fig.savefig(path)
            return
        except PermissionError:
            time.sleep(0.8)
    fig.savefig(path)


def _load_report() -> dict:
    return json.loads((config.OUTPUT_DIR / "evaluation_report.json").read_text(encoding="utf-8"))


# ---------------- 01 核心指标对比 ------------------------------------------------
def fig_metrics(rep: dict) -> None:
    s = rep["summary"]
    n = s["total"]
    items = [
        ("答案命中率\n(Accuracy)", s["baseline_answer_hit"] / n * 100, s["optimized_answer_hit"] / n * 100),
        ("检索 Top-1 命中率", s["baseline_top1_hit"] / n * 100, s["optimized_top1_hit"] / n * 100),
        ("检索 MRR\n(×100)", s["baseline_mrr"] * 100, s["optimized_mrr"] * 100),
        ("平均响应时间(s)\n(越低越好)", s["baseline_avg_elapsed"], s["optimized_avg_elapsed"]),
    ]
    labels = [i[0] for i in items]
    base = [i[1] for i in items]
    opt = [i[2] for i in items]

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=160)
    x = range(len(items))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], base, w, label="优化前（基线）", color=C_BASE)
    b2 = ax.bar([i + w / 2 for i in x], opt, w, label="优化后", color=C_OPT)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=10, color=C_TEXT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("数值", fontsize=11)
    ax.set_title("优化前后核心指标对比（10 题）", fontsize=14, fontweight="bold", color=C_TEXT, pad=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", color=C_GRID, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "01_metrics_before_after.png")
    plt.close(fig)


# ---------------- 02 逐题命中对比 ------------------------------------------------
def fig_per_question(rep: dict) -> None:
    cases = rep["cases"]
    ids = [f"Q{c['id']}" for c in cases]
    b = [1 if c["baseline"]["answer_hit"] else 0 for c in cases]
    o = [1 if c["optimized"]["answer_hit"] else 0 for c in cases]

    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=160)
    x = range(len(cases))
    w = 0.36
    ax.bar([i - w / 2 for i in x], b, w, label="优化前答案命中", color=C_BASE)
    ax.bar([i + w / 2 for i in x], o, w, label="优化后答案命中", color=C_OPT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["未命中", "命中"], fontsize=10)
    ax.set_ylim(0, 1.35)
    ax.set_title("逐题答案命中对比", fontsize=14, fontweight="bold", color=C_TEXT, pad=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(axis="y", color=C_GRID, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "02_per_question_hit.png")
    plt.close(fig)


# ---------------- 03 逐题耗时对比 ------------------------------------------------
def fig_latency(rep: dict) -> None:
    cases = rep["cases"]
    ids = [f"Q{c['id']}" for c in cases]
    b = [c["baseline"]["elapsed"] for c in cases]
    o = [c["optimized"]["elapsed"] for c in cases]

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    x = range(len(cases))
    ax.plot(x, b, "o-", color=C_BASE, label="优化前耗时", linewidth=2, markersize=6)
    ax.plot(x, o, "s-", color=C_OPT, label="优化后耗时", linewidth=2, markersize=6)
    ax.axhline(3.0, color="#dc2626", linestyle="--", linewidth=1.4)
    ax.annotate("验收上限 3 s", (len(cases) - 0.6, 3.0), color="#dc2626", fontsize=10,
                ha="right", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, fontsize=10)
    ax.set_ylabel("响应时间 (s)", fontsize=11)
    ax.set_title("逐题响应时间对比（含查询向量化）", fontsize=14, fontweight="bold",
                 color=C_TEXT, pad=14)
    ax.legend(fontsize=10)
    ax.grid(color=C_GRID, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "03_latency_compare.png")
    plt.close(fig)


# ---------------- 通用：流程图 ----------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec="#ffffff", fs=10, tc="#111827"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.5)


def _arrow(ax, p1, p2, color="#6b7280"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=color, zorder=1,
                                 shrinkA=1, shrinkB=1))


# ---------------- 04 优化后链路流程图 --------------------------------------------
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4), dpi=160)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(6.5, 6.05, "优化后 RAG 问答链路", ha="center", fontsize=15,
            fontweight="bold", color=C_TEXT)

    # 离线：建库
    ax.text(0.15, 5.35, "离线建库", fontsize=11, fontweight="bold", color="#374151")
    _box(ax, 0.15, 4.35, 2.1, 0.75, "PDF 解析优化\n版式清洗 / 表格结构化", "#dbeafe")
    _box(ax, 2.55, 4.35, 2.1, 0.75, "结构感知分块\n小节切分 / 表格原子块", "#dbeafe")
    _box(ax, 4.95, 4.35, 2.1, 0.75, "父子块 + 向量化\nbge-m3 嵌入", "#dbeafe")
    _box(ax, 7.35, 4.35, 2.1, 0.75, "FAISS 向量库\n+ BM25 全局语料", "#dbeafe")
    _arrow(ax, (2.25, 4.72), (2.55, 4.72))
    _arrow(ax, (4.65, 4.72), (4.95, 4.72))
    _arrow(ax, (7.05, 4.72), (7.35, 4.72))

    # 在线：问答
    ax.text(0.15, 3.35, "在线问答", fontsize=11, fontweight="bold", color="#374151")
    _box(ax, 0.15, 2.35, 2.1, 0.75, "用户提问\n（中文 / English）", "#dcfce7")
    _box(ax, 2.55, 2.35, 2.1, 0.75, "Query 理解\n噪声剥离 / 关键词 / 类型", "#dcfce7")
    _box(ax, 4.95, 2.35, 2.1, 0.75, "多路查询召回\n向量 + BM25 双路", "#dcfce7")
    _box(ax, 7.35, 2.35, 2.1, 0.75, "RRF 融合\n+ 加权线性重排", "#dcfce7")
    _box(ax, 9.75, 2.35, 3.1, 0.75, "抽取式答案合成\n类型感知打分 / 页码引用", "#dcfce7")
    _arrow(ax, (2.25, 2.72), (2.55, 2.72))
    _arrow(ax, (4.65, 2.72), (4.95, 2.72))
    _arrow(ax, (7.05, 2.72), (7.35, 2.72))
    _arrow(ax, (9.45, 2.72), (9.75, 2.72))

    # 离线 → 在线
    _arrow(ax, (8.4, 4.35), (8.4, 3.1), color="#2563eb")
    ax.text(8.55, 3.65, "索引\n复用", fontsize=9, color="#2563eb")

    # 输出
    _box(ax, 9.75, 0.95, 3.1, 0.85, "答案 + 引用页码 + 证据句\n（零幻觉、可溯源）", "#fef3c7", fs=10)
    _arrow(ax, (11.3, 2.35), (11.3, 1.8), color="#d97706")

    ax.text(0.15, 1.4, "验收：准确率 ≥ 90%（实测 100%） ｜ 响应 ≤ 3 s（实测平均 1.12 s）",
            fontsize=10.5, color="#374151")

    fig.tight_layout()
    _save(fig, "04_pipeline_flow.png")
    plt.close(fig)


# ---------------- 05 分块策略对比 ------------------------------------------------
def fig_chunking() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=160)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(6.25, 4.85, "分块策略：优化前 vs 优化后", ha="center", fontsize=14,
            fontweight="bold", color=C_TEXT)

    # 优化前
    ax.text(0.2, 4.3, "优化前：固定 500/80 字符递归切片", fontsize=11, fontweight="bold",
            color=C_BASE)
    xs = 0.2
    for i, w in enumerate([1.6, 1.9, 1.4, 1.8, 1.6]):
        _box(ax, xs, 3.35, w, 0.6, f"块{i + 1}", "#ffedd5", fs=9)
        xs += w + 0.12
    ax.text(0.2, 2.95, "问题：切在句中 / 表头与数据分离 / 跨小节串味 / 上下文不足",
            fontsize=9.5, color="#b45309")

    # 优化后
    ax.text(0.2, 2.35, "优化后：结构感知分块 + 表格原子块 + 父子块", fontsize=11,
            fontweight="bold", color=C_OPT)
    _box(ax, 0.2, 1.25, 2.6, 0.85, "小节 A 标题\n（父块）", "#dcfce7", fs=9)
    _box(ax, 2.95, 1.25, 1.5, 0.85, "子块 a1", "#bbf7d0", fs=9)
    _box(ax, 4.55, 1.25, 1.5, 0.85, "子块 a2", "#bbf7d0", fs=9)
    _box(ax, 6.2, 1.25, 1.9, 0.85, "表格原子块\n(不切分)", "#fde68a", fs=9)
    _box(ax, 8.25, 1.25, 2.6, 0.85, "小节 B 标题\n（父块）", "#dcfce7", fs=9)
    _box(ax, 11.0, 1.25, 1.3, 0.85, "子块 b1", "#bbf7d0", fs=9)
    ax.text(0.2, 0.7, "效果：检索用子块（精准）+ 答案抽取用父块（上下文完整），"
                      "表格数值完整保留", fontsize=9.5, color="#15803d")

    fig.tight_layout()
    _save(fig, "05_chunking_compare.png")
    plt.close(fig)


# ---------------- 06 四层优化总览 ------------------------------------------------
def fig_layers() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=160)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(6.5, 5.25, "四层优化点总览", ha="center", fontsize=15, fontweight="bold", color=C_TEXT)

    data = [
        ("① PDF 解析", "#dbeafe", ["版式清洗：剔除页眉/页脚/水印",
                                    "表格结构化：Markdown 渲染",
                                    "行合并与空白规整",
                                    "章节标题识别"]),
        ("② 分块", "#dcfce7", ["结构感知：按标题切语义小节",
                                "表格原子块：整表不切分",
                                "父子块：检索准 + 上下文全",
                                "过短合并 / 超长切分"]),
        ("③ 检索", "#fef3c7", ["多路查询：核心句/关键词/实体",
                                "双路召回：向量 + BM25",
                                "RRF 融合候选",
                                "加权线性重排 + 父块去重"]),
        ("④ 答案合成", "#fce7f3", ["抽取式：零幻觉、可溯源",
                                    "类型感知打分（数值/比例/列举）",
                                    "对比词 / 套话惩罚",
                                    "毫秒级返回，附页码引用"]),
    ]
    w, gap = 2.95, 0.2
    for i, (title, color, bullets) in enumerate(data):
        x = 0.3 + i * (w + gap)
        _box(ax, x, 4.0, w, 0.62, title, color, fs=12)
        for j, b in enumerate(bullets):
            _box(ax, x, 3.28 - j * 0.72, w, 0.6, b, "#f9fafb", ec="#e5e7eb", fs=9)
        if i < len(data) - 1:
            _arrow(ax, (x + w + 0.02, 4.31), (x + w + gap - 0.02, 4.31), color="#9ca3af")

    ax.text(0.3, 0.35, "优化前基线：固定切片 + 单路查询 + RRF 排名 + 本地 LLM 生成（准确率 60%、平均 3.53 s）",
            fontsize=10, color="#b45309")
    ax.text(0.3, 0.05, "优化后：四层协同 → 准确率 100%、平均 1.12 s",
            fontsize=10, color="#15803d")

    fig.tight_layout()
    _save(fig, "06_optimization_layers.png")
    plt.close(fig)


_ALL = {
    "01": ("01_metrics_before_after.png", lambda rep: fig_metrics(rep)),
    "02": ("02_per_question_hit.png", lambda rep: fig_per_question(rep)),
    "03": ("03_latency_compare.png", lambda rep: fig_latency(rep)),
    "04": ("04_pipeline_flow.png", lambda rep: fig_pipeline()),
    "05": ("05_chunking_compare.png", lambda rep: fig_chunking()),
    "06": ("06_optimization_layers.png", lambda rep: fig_layers()),
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="生成优化对比图表与流程图")
    parser.add_argument("--only", default="", help="仅生成指定编号的图（如 02；默认全部）")
    args = parser.parse_args()

    rep = _load_report()
    keys = [args.only] if args.only else list(_ALL)
    for k in keys:
        if k not in _ALL:
            print(f"跳过未知编号：{k}")
            continue
        name, fn = _ALL[k]
        fn(rep)
        print(f"已生成 {name}")
    print("图表目录：", _fig_dir())


if __name__ == "__main__":
    main()
