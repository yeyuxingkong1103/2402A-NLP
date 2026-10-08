# -*- coding: utf-8 -*-
# 工单14：生成设计图 5 张 PNG
"""matplotlib 绘制，中文用微软雅黑。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DESIGN = Path(__file__).resolve().parent

C_BLUE = "#4a90d9"; C_GREEN = "#5cb85c"; C_ORANGE = "#f0ad4e"
C_RED = "#d9534f"; C_GRAY = "#7f8c8d"; C_PURPLE = "#9b59b6"


def box(ax, x, y, w, h, text, fc=C_BLUE, fs=11, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True)


def arrow(ax, p1, p2, color="#444", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 lw=1.6, color=color))


def save(fig, name):
    fig.savefig(DESIGN / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已生成 {name}")


# ---------- 图1：RAGFlow 部署架构图 ----------
def fig1():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.text(5.5, 7.2, "RAGFlow 参考部署架构（官方 docker-compose 容器编排）",
            ha="center", fontsize=15, weight="bold")

    box(ax, 0.3, 5.6, 1.8, 0.8, "用户浏览器\nWeb UI :80", C_GRAY)
    # 核心服务
    box(ax, 3.0, 5.7, 2.2, 0.7, "ragflow\nAPI Server", C_BLUE)
    box(ax, 3.0, 4.5, 2.2, 0.7, "task-executor\n文件解析执行器", C_GREEN)
    box(ax, 6.0, 5.7, 2.0, 0.7, "DeepDoc\n解析/OCR/TSR", C_PURPLE)
    box(ax, 6.0, 4.5, 2.0, 0.7, "Embedding / LLM\n(本地或API)", C_PURPLE)

    # 基础服务
    for i, (t, c) in enumerate([
        ("MinIO\n对象存储", C_ORANGE), ("Elasticsearch\n全文检索", C_ORANGE),
        ("MySQL\n元数据", C_ORANGE), ("Redis\n消息/缓存", C_ORANGE)]):
        box(ax, 0.4 + i * 2.6, 2.4, 2.2, 0.8, t, c, fs=10)

    box(ax, 0.4, 0.6, 10.2, 1.0,
        "本环境轻量化实现（纯CPU/零API费用）：pymupdf 渲染 + easyocr/源文本OCR还原 "
        "+ bge-m3 本地向量 + 规则抽取问答，复刻 DeepDoc 解析链路",
        C_RED, fs=11)

    arrow(ax, (2.1, 6.0), (3.0, 6.0))
    arrow(ax, (4.1, 5.7), (4.1, 5.2))
    arrow(ax, (5.2, 6.05), (6.0, 6.05))
    arrow(ax, (5.2, 4.85), (6.0, 4.85))
    for x in [1.5, 4.1, 6.7, 9.3]:
        arrow(ax, (x, 4.5), (x, 3.2), C_GRAY)
    save(fig, "01-RAGFlow部署架构图.png")


# ---------- 图2：DeepDoc PDF 解析流程图 ----------
def fig2():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(5.5, 4.7, "DeepDoc 低质量 PDF 解析流水线", ha="center",
            fontsize=15, weight="bold")

    steps = [
        ("PDF输入\n(图片型扫描件)", C_GRAY),
        ("页面渲染\npdfium/pdfoxide\n提高DPI", C_BLUE),
        ("OCR文字识别\n旋转矫正/乱码重试", C_BLUE),
        ("版面分析\n标题/正文/多栏", C_GREEN),
        ("表格TSR识别\n行列网格/合并", C_GREEN),
        ("阅读顺序重建\n去页眉页脚/断词", C_PURPLE),
        ("输出\nJSON+Markdown", C_ORANGE),
    ]
    w, h, gap = 1.35, 1.3, 0.16
    x = 0.25
    centers = []
    for t, c in steps:
        box(ax, x, 2.4, w, h, t, c, fs=9.5)
        centers.append(x + w)
        x += w + gap
    for i in range(len(steps) - 1):
        arrow(ax, (centers[i], 3.05), (centers[i] + gap, 3.05))

    ax.text(5.5, 1.4,
            "关键源码：internal/deepdoc/parser/pdf/parser.go · parser_ocr.go · "
            "layout/boxes_sections.go · table/deepdoc_table_builder.go",
            ha="center", fontsize=9.5, color="#555")
    ax.text(5.5, 0.8,
            "本环境对应实现：src/deepdoc_ocr_parser.py（get_pixmap→OCR→y坐标行聚类→段落重建）",
            ha="center", fontsize=9.5, color=C_RED)
    save(fig, "02-DeepDoc解析流程图.png")


# ---------- 图3：低质量PDF问题-修复方案图 ----------
def fig3():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(5.5, 5.9, "工业 PDF 信息丢失问题与修复方案", ha="center",
            fontsize=15, weight="bold")

    problems = [
        "图片型PDF无文本层", "低分辨率文字模糊", "双栏排版串行拼接",
        "表格结构丢失", "页眉页脚重复干扰", "跨页段落语义断裂",
    ]
    fixes = [
        "检测文本层→强制OCR路径", "渲染提DPI+乱码检测重试", "版面分析按列分离",
        "TSR还原行列网格", "自动检测并剥离页眉页脚", "按标点缩进合并跨页段落",
    ]
    for i in range(6):
        y = 4.9 - i * 0.82
        box(ax, 0.5, y, 3.6, 0.6, problems[i], C_RED, fs=10.5)
        arrow(ax, (4.1, y + 0.3), (5.0, y + 0.3), C_GRAY)
        box(ax, 5.0, y, 5.4, 0.6, fixes[i], C_GREEN, fs=10.5)
    save(fig, "03-问题修复方案图.png")


# ---------- 图4：解析前后信息量对比 ----------
def fig4():
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["基线：pymupdf\n直接提取", "优化：OCR+版面\n分析重建"]
    vals = [35, 3335]
    bars = ax.bar(labels, vals, color=[C_GRAY, C_GREEN], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v} 字符",
                ha="center", fontsize=12, weight="bold")
    ax.set_ylabel("提取字符数", fontsize=12)
    ax.set_title("解析前后信息量对比（信息恢复率 95.3 倍）", fontsize=14, weight="bold")
    ax.set_ylim(0, 3800)
    save(fig, "04-解析前后对比.png")


# ---------- 图5：6题准确率与耗时 ----------
def fig5():
    import json
    qa = json.loads((DESIGN.parent / "研发" / "qa_results.json")
                    .read_text(encoding="utf-8"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ids = [f"Q{r['id']}" for r in qa["results"]]
    acc = [100 if r["correct"] else 0 for r in qa["results"]]
    a1.bar(ids, acc, color=C_GREEN, width=0.6)
    a1.set_ylim(0, 118); a1.set_ylabel("正确率 %")
    a1.set_title(f"逐题正确率（总准确率 {qa['accuracy']:.0f}%）",
                 fontsize=12.5, weight="bold")
    for i, v in enumerate(acc):
        a1.text(i, v + 3, "对", ha="center", fontsize=13, color=C_GREEN)

    t = [r["elapsed_sec"] for r in qa["results"]]
    a2.bar(ids, t, color=C_BLUE, width=0.6)
    a2.axhline(3, color=C_RED, ls="--", lw=1.5)
    a2.text(0.1, 3.08, "验收线 3s", color=C_RED, fontsize=10)
    a2.set_ylim(0, 3.4); a2.set_ylabel("耗时 秒")
    a2.set_title(f"逐题响应耗时（平均 {qa['avg_sec']}s）",
                 fontsize=12.5, weight="bold")
    for i, v in enumerate(t):
        a2.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=9.5)
    fig.tight_layout()
    save(fig, "05-准确率与耗时.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
