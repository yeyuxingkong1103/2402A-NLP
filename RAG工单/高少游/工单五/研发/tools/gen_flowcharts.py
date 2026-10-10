# -*- coding: utf-8 -*-
"""流程图 / 架构图生成脚本。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

生成（输出到 设计/assets/）：
    01_系统总体架构.png        —— 分层架构（数据 / 索引 / 检索 / 多轮 / 应用）
    02_多轮对话问答流程.png    —— 一次问答的端到端流程
    03_Query改写决策流程.png   —— 指代消解 / 省略补全的判定与改写流程
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent.parent / "设计" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = ["#e3f2fd", "#e8f5e9", "#fff3e0", "#f3e5f5", "#e0f7fa", "#fce4ec"]
EDGE = "#37474f"


def box(ax, x, y, w, h, text, fc="#e3f2fd", fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.2, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p1, p2, color=EDGE, style="-|>", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 color=color, linewidth=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w=12, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- 1 架构图
def arch():
    fig, ax = canvas(12, 8.5)
    ax.text(5, 9.7, "招股说明书多轮检索问答系统 · 总体架构", ha="center",
            fontsize=15, fontweight="bold")
    layers = [
        ("应用层", ["Streamlit 多轮对话界面", "命令行演示 run_demo", "评测 evaluate_multiturn"]),
        ("多轮对话层（本工单核心）", ["会话状态 Conversation", "Query 改写 QueryRewriter",
                                 "指代消解 / 省略补全"]),
        ("检索层", ["混合检索 向量+BM25+RRF", "多信号重排 Reranker", "文档级路由（公司消歧）"]),
        ("理解层", ["Query 理解（关键词/实体/答案类型）", "答案合成 AnswerBuilder"]),
        ("索引层", ["切片 Chunking", "Ollama bge-m3 向量", "BM25 倒排"]),
        ("数据层", ["招股说明书1（兴图新科）", "招股说明书2（力源信息）"]),
    ]
    y = 8.4
    for i, (name, items) in enumerate(layers):
        h = 1.05
        box(ax, 0.3, y - h, 2.0, h, name, fc=PALETTE[i % len(PALETTE)], fs=11, bold=True)
        n = len(items)
        gap = 0.12
        w = (7.3 - gap * (n - 1)) / n
        x = 2.5
        for it in items:
            box(ax, x, y - h, w, h, it, fc="#ffffff", fs=9)
            x += w + gap
        if i < len(layers) - 1:
            arrow(ax, (5.1, y - h), (5.1, y - h - 0.22))
        y -= h + 0.28
    fig.tight_layout()
    fig.savefig(OUT / "01_系统总体架构.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- 2 多轮问答流程
def flow_qa():
    fig, ax = canvas(10, 11)
    ax.text(5, 10.5, "多轮对话问答流程", ha="center", fontsize=15, fontweight="bold")
    steps = [
        ("用户提问（含指代 / 省略）", "#e3f2fd"),
        ("读取会话上下文\n（话题实体 / 活跃文档 / 上一轮意图）", "#e8f5e9"),
        ("Query 改写\n（指代消解 + 省略补全）→ 独立问句", "#fff3e0"),
        ("Query 理解\n（关键词 / 实体 / 答案类型 / 文档路由）", "#f3e5f5"),
        ("混合检索\n向量召回 + BM25 召回 + RRF 融合", "#e0f7fa"),
        ("多信号重排\n（含多轮话题实体加成）", "#fce4ec"),
        ("抽取式答案合成 + 证据溯源", "#e3f2fd"),
        ("写回会话记忆（Turn）→ 返回答案", "#e8f5e9"),
    ]
    y = 9.6
    h = 0.86
    for i, (txt, fc) in enumerate(steps):
        box(ax, 2.2, y - h, 5.6, h, txt, fc=fc, fs=11)
        if i < len(steps) - 1:
            arrow(ax, (5.0, y - h), (5.0, y - h - 0.26))
        y -= h + 0.26
    fig.tight_layout()
    fig.savefig(OUT / "02_多轮对话问答流程.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- 3 Query 改写流程
def flow_rewrite():
    fig, ax = canvas(11, 8.5)
    ax.text(5, 9.6, "多轮 Query 改写决策流程", ha="center", fontsize=15, fontweight="bold")
    box(ax, 3.9, 8.4, 2.4, 0.7, "输入用户问句", fc="#e3f2fd", fs=11)
    box(ax, 3.4, 7.2, 3.4, 0.75, "命中省略式追问？\n（那 X 呢？/ 呢？）", fc="#fff3e0", fs=10)
    arrow(ax, (5.1, 8.4), (5.1, 7.95))
    box(ax, 7.1, 7.2, 2.6, 0.75, "继承上一轮意图骨架\n替换/补全主语实体", fc="#e8f5e9", fs=10)
    arrow(ax, (6.8, 7.57), (7.1, 7.57))
    ax.text(6.95, 7.72, "是", fontsize=10, color="#c62828")
    box(ax, 0.3, 7.2, 2.7, 0.75, "命中指代词？\n（他 / 这个公司…）", fc="#fff3e0", fs=10)
    arrow(ax, (3.4, 7.57), (3.0, 7.57))
    ax.text(3.05, 7.72, "否", fontsize=10, color="#2e7d32")
    box(ax, 0.3, 5.9, 2.7, 0.75, "指代词 → 替换为\n当前话题公司全名", fc="#e8f5e9", fs=10)
    arrow(ax, (1.65, 7.2), (1.65, 6.65))
    box(ax, 3.9, 5.9, 2.4, 0.75, "缺主语追问？", fc="#fff3e0", fs=10)
    arrow(ax, (3.0, 7.57), (3.9, 6.65), rad=-0.2)
    box(ax, 7.1, 5.9, 2.6, 0.75, "注入话题公司 + 意图", fc="#e8f5e9", fs=10)
    arrow(ax, (6.3, 6.27), (7.1, 6.27))
    ax.text(6.55, 6.42, "是", fontsize=10, color="#c62828")
    box(ax, 3.9, 4.5, 2.4, 0.75, "独立问句（不改写）", fc="#e0f7fa", fs=10)
    arrow(ax, (5.1, 5.9), (5.1, 5.25))
    ax.text(5.25, 5.5, "否", fontsize=10, color="#2e7d32")
    box(ax, 3.2, 3.0, 3.8, 0.8, "输出：独立问句 + 文档路由提示 + 继承实体", fc="#f3e5f5", fs=11, bold=True)
    for xx in (1.65, 8.4, 5.1):
        arrow(ax, (xx, 5.9), (5.1, 3.8), rad=0.0)
    arrow(ax, (1.65, 5.9), (3.2, 3.4), rad=0.25)
    arrow(ax, (8.4, 5.9), (7.0, 3.4), rad=-0.25)
    ax.text(5, 2.2, "示例：\nQ2「他参与的哪个工程…」→「武汉兴图新科电子股份有限公司参与的哪个工程…」\n"
                    "Q4「那武汉力源信息技术股份有限公司呢？」→「武汉力源信息技术股份有限公司法定代表人是谁」",
            ha="center", fontsize=9.5, color="#455a64")
    fig.tight_layout()
    fig.savefig(OUT / "03_Query改写决策流程.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    arch()
    flow_qa()
    flow_rewrite()
    print("流程图已生成 →", OUT)