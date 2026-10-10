# -*- coding: utf-8 -*-
"""设计文档流程图生成脚本
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

用 matplotlib 绘制三张流程/架构图（PNG），供 设计/assets 引用：
    1. 系统总体架构.png       —— 离线建库 + 在线问答 两条链路
    2. 图像内容解析流程.png   —— 图形抽取 → 语义解析 → 编码入库
    3. 检索与答案合成流程.png —— Query 理解 → 多路召回 → 重排 → 抽取式合成

用法：
    python tools/gen_flowcharts.py --out "../设计/assets"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配色
C_OFF = "#dbeafe"     # 离线（蓝）
C_OFF_E = "#2563eb"
C_ON = "#dcfce7"      # 在线（绿）
C_ON_E = "#16a34a"
C_STORE = "#fef3c7"   # 存储（黄）
C_STORE_E = "#d97706"
C_CORE = "#ede9fe"    # 核心（紫）
C_CORE_E = "#7c3aed"
C_TEXT = "#1e293b"


def box(ax, x, y, w, h, text, fc=C_OFF, ec=C_OFF_E, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=C_TEXT, zorder=3, fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, p1, p2, color="#475569", style="-|>", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def legend(ax, items, x=0.02, y=0.02):
    for i, (c, e, label) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x, y + i * 0.045), 0.018, 0.028,
                                    boxstyle="round,pad=0.004", linewidth=1.2,
                                    edgecolor=e, facecolor=c))
        ax.text(x + 0.026, y + i * 0.045 + 0.014, label, va="center", fontsize=9.5,
                color=C_TEXT)


# ---------------- 图 1：系统总体架构 ------------------------------------------
def chart_architecture(out: Path) -> None:
    fig, ax = new_ax(12.4, 8.2)
    ax.text(0.5, 0.965, "招股说明书图像内容解析问答系统 · 总体架构", ha="center",
            fontsize=17, fontweight="bold", color="#0f172a")
    ax.text(0.5, 0.925, "工单编号：人工智能 NLP-RAG-图像内容解析及检索优化",
            ha="center", fontsize=10.5, color="#64748b")

    # 分区背景
    ax.add_patch(FancyBboxPatch((0.015, 0.30), 0.47, 0.575, boxstyle="round,pad=0.008",
                                linewidth=1.4, edgecolor="#93c5fd", facecolor="#f8fbff",
                                linestyle="--", zorder=0))
    ax.text(0.25, 0.855, "离线：知识库构建  build_kb.py", ha="center", fontsize=12.5,
            fontweight="bold", color=C_OFF_E)
    ax.add_patch(FancyBboxPatch((0.505, 0.30), 0.48, 0.575, boxstyle="round,pad=0.008",
                                linewidth=1.4, edgecolor="#86efac", facecolor="#f7fff9",
                                linestyle="--", zorder=0))
    ax.text(0.745, 0.855, "在线：问答服务  app.py / evaluate.py", ha="center",
            fontsize=12.5, fontweight="bold", color=C_ON_E)

    # ---- 离线列 ----
    box(ax, 0.06, 0.755, 0.37, 0.065, "PDF 文档：招股说明书1 / 2", C_OFF, C_OFF_E, 11.5, True)
    box(ax, 0.035, 0.635, 0.20, 0.075, "文本 / 表格解析\npdf_parser + table_parser")
    box(ax, 0.255, 0.635, 0.20, 0.075, "图形区域抽取\nfigure_extractor")
    box(ax, 0.035, 0.515, 0.20, 0.075, "结构感知分块\nchunking（父子块）")
    box(ax, 0.255, 0.515, 0.20, 0.075, "图像语义解析\nfigure_semantics + OCR",
        C_CORE, C_CORE_E)
    box(ax, 0.035, 0.395, 0.20, 0.075, "文本 / 表格 /\n图形语义块")
    box(ax, 0.255, 0.395, 0.20, 0.075, "CLIP 图像编码\nimage_encoder")
    box(ax, 0.06, 0.305, 0.37, 0.06, "优化后 FAISS 文本向量库  +  CLIP 图像向量库",
        C_STORE, C_STORE_E, 11, True)

    arrow(ax, (0.245, 0.755), (0.135, 0.712))
    arrow(ax, (0.245, 0.755), (0.355, 0.712))
    arrow(ax, (0.135, 0.635), (0.135, 0.592))
    arrow(ax, (0.355, 0.635), (0.355, 0.592))
    arrow(ax, (0.135, 0.515), (0.135, 0.472))
    arrow(ax, (0.355, 0.515), (0.355, 0.472))
    arrow(ax, (0.135, 0.395), (0.16, 0.367))
    arrow(ax, (0.355, 0.395), (0.33, 0.367))

    # ---- 在线列 ----
    box(ax, 0.545, 0.755, 0.40, 0.065, "用户提问（中文 / 英文）", C_ON, C_ON_E, 11.5, True)
    box(ax, 0.545, 0.635, 0.40, 0.075, "Query 理解：核心问句 / 关键词 / 实体 / 文档路由",
        C_CORE, C_CORE_E)
    box(ax, 0.545, 0.515, 0.40, 0.075, "多路召回：向量 Top30 + BM25 Top30 + 以文搜图 Top5")
    box(ax, 0.545, 0.395, 0.40, 0.075, "多信号重排：关键词 / 数值 / 表格 / 图形 / CLIP")
    box(ax, 0.545, 0.305, 0.40, 0.065, "抽取式答案合成：答案 + 来源页码 + 响应耗时",
        C_ON, C_ON_E, 11, True)

    arrow(ax, (0.745, 0.755), (0.745, 0.712))
    arrow(ax, (0.745, 0.635), (0.745, 0.592))
    arrow(ax, (0.745, 0.515), (0.745, 0.472))
    arrow(ax, (0.745, 0.395), (0.745, 0.372))

    # 离线 → 在线
    arrow(ax, (0.43, 0.335), (0.545, 0.55), color="#7c3aed", lw=2.2, rad=-0.15)
    arrow(ax, (0.43, 0.335), (0.545, 0.335), color="#7c3aed", lw=2.2, rad=-0.1)
    ax.text(0.487, 0.44, "向量库\n加载", ha="center", fontsize=9, color="#7c3aed")

    legend(ax, [(C_OFF, C_OFF_E, "解析 / 分块"), (C_CORE, C_CORE_E, "语义理解"),
                (C_STORE, C_STORE_E, "向量存储"), (C_ON, C_ON_E, "在线服务")])
    fig.tight_layout()
    fig.savefig(out / "01_系统总体架构.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- 图 2：图像内容解析流程 --------------------------------------
def chart_figure(out: Path) -> None:
    fig, ax = new_ax(12.0, 6.4)
    ax.text(0.5, 0.955, "图像内容解析流程（本工单核心）", ha="center", fontsize=16.5,
            fontweight="bold", color="#0f172a")

    # 主链
    box(ax, 0.03, 0.80, 0.16, 0.085, "PDF 页面", C_OFF, C_OFF_E, 11.5, True)
    box(ax, 0.23, 0.80, 0.20, 0.085, "图形区域抽取\n位图 + 矢量簇", C_OFF, C_OFF_E)
    box(ax, 0.47, 0.80, 0.20, 0.085, "类型判定\n矩形+连线 / 坐标轴", C_CORE, C_CORE_E)
    box(ax, 0.71, 0.80, 0.26, 0.085, "语义解析\n组织结构 / 统计图", C_CORE, C_CORE_E)
    box(ax, 0.30, 0.585, 0.40, 0.085, "生成图像语义文本块（ctype=figure）", C_STORE, C_STORE_E, 11.5, True)
    box(ax, 0.30, 0.40, 0.40, 0.085, "CLIP 图像编码 → 图像向量库（离线预计算）", C_STORE, C_STORE_E, 11.5, True)

    arrow(ax, (0.19, 0.8425), (0.23, 0.8425))
    arrow(ax, (0.43, 0.8425), (0.47, 0.8425))
    arrow(ax, (0.67, 0.8425), (0.71, 0.8425))
    arrow(ax, (0.84, 0.80), (0.60, 0.67), rad=0.12)
    arrow(ax, (0.50, 0.585), (0.50, 0.485))
    arrow(ax, (0.50, 0.40), (0.50, 0.30), color="#d97706")

    # 组织结构分支
    box(ax, 0.045, 0.44, 0.235, 0.075, "组织结构图\n节点框 + 连接线图", C_CORE, C_CORE_E, 10.5)
    box(ax, 0.045, 0.335, 0.235, 0.075, "母线归属法\n还原有向层级树", "#fce7f3", "#db2777", 10.5)
    box(ax, 0.045, 0.23, 0.235, 0.075, "输出：部门 / 处室数量与层级", "#f1f5f9", "#64748b", 10.5)
    arrow(ax, (0.71, 0.8425), (0.163, 0.515), rad=0.18)
    arrow(ax, (0.163, 0.44), (0.163, 0.41))
    arrow(ax, (0.163, 0.335), (0.163, 0.305))

    # 统计图分支
    box(ax, 0.72, 0.44, 0.245, 0.075, "统计图\n渲染高清位图 + OCR", C_CORE, C_CORE_E, 10.5)
    box(ax, 0.72, 0.335, 0.245, 0.075, "去水印前处理\n标签—数值配对", "#fce7f3", "#db2777", 10.5)
    box(ax, 0.72, 0.23, 0.245, 0.075, "输出：行业增速 / 涨跌幅", "#f1f5f9", "#64748b", 10.5)
    arrow(ax, (0.84, 0.80), (0.8425, 0.515), rad=-0.18)
    arrow(ax, (0.8425, 0.44), (0.8425, 0.41))
    arrow(ax, (0.8425, 0.335), (0.8425, 0.305))

    legend(ax, [(C_OFF, C_OFF_E, "区域抽取"), (C_CORE, C_CORE_E, "语义解析"),
                ("#fce7f3", "#db2777", "关键算法"), (C_STORE, C_STORE_E, "入库")],
           x=0.03, y=0.04)
    fig.tight_layout()
    fig.savefig(out / "02_图像内容解析流程.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- 图 3：检索与答案合成流程 ------------------------------------
def chart_retrieval(out: Path) -> None:
    fig, ax = new_ax(12.0, 7.0)
    ax.text(0.5, 0.96, "检索与答案合成流程（在线）", ha="center", fontsize=16.5,
            fontweight="bold", color="#0f172a")

    box(ax, 0.33, 0.845, 0.34, 0.075, "用户提问", C_ON, C_ON_E, 12, True)
    box(ax, 0.33, 0.715, 0.34, 0.08, "Query 理解（确定性规则）\n核心问句 / 关键词 / 实体 / 题型 / 文档路由",
        C_CORE, C_CORE_E, 10.5)

    # 三路召回
    box(ax, 0.04, 0.545, 0.27, 0.085, "向量召回 Top30\n（bge-m3 语义）", C_OFF, C_OFF_E, 10.5)
    box(ax, 0.365, 0.545, 0.27, 0.085, "BM25 召回 Top30\n（词法精确）", C_OFF, C_OFF_E, 10.5)
    box(ax, 0.69, 0.545, 0.27, 0.085, "以文搜图 Top5\n（CLIP 跨模态）", C_CORE, C_CORE_E, 10.5)

    arrow(ax, (0.50, 0.715), (0.175, 0.63), rad=0.1)
    arrow(ax, (0.50, 0.715), (0.50, 0.63))
    arrow(ax, (0.50, 0.715), (0.825, 0.63), rad=-0.1)

    box(ax, 0.24, 0.395, 0.52, 0.075, "RRF 融合 + 候选池合并（含图形语义块注入）",
        C_STORE, C_STORE_E, 11)
    arrow(ax, (0.175, 0.545), (0.42, 0.47), rad=0.1)
    arrow(ax, (0.50, 0.545), (0.50, 0.47))
    arrow(ax, (0.825, 0.545), (0.58, 0.47), rad=-0.1)

    box(ax, 0.24, 0.27, 0.52, 0.075, "文档级路由消歧（公司别名 → 源文件）", "#e0f2fe", "#0284c7", 11)
    arrow(ax, (0.50, 0.395), (0.50, 0.345))

    box(ax, 0.24, 0.145, 0.52, 0.08, "多信号重排\n向量/BM25/关键词/数值/实体/表格/图形/CLIP/文档",
        C_CORE, C_CORE_E, 10.5)
    arrow(ax, (0.50, 0.27), (0.50, 0.225))

    box(ax, 0.24, 0.03, 0.52, 0.075, "抽取式答案合成 → 答案 + 来源页码 + 耗时（≤3s）",
        C_ON, C_ON_E, 11, True)
    arrow(ax, (0.50, 0.145), (0.50, 0.105))

    legend(ax, [(C_ON, C_ON_E, "输入/输出"), (C_CORE, C_CORE_E, "理解/跨模态"),
                (C_OFF, C_OFF_E, "文本召回"), (C_STORE, C_STORE_E, "融合")],
           x=0.03, y=0.72)
    fig.tight_layout()
    fig.savefig(out / "03_检索与答案合成流程.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../设计/assets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chart_architecture(out)
    chart_figure(out)
    chart_retrieval(out)
    print(f"流程图已生成 → {out.resolve()}")


if __name__ == "__main__":
    main()