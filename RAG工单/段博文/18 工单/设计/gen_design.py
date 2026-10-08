# -*- coding: utf-8 -*-
"""生成工单18设计图 5 张 PNG。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GRAY, C_DARK = (
    "#4a90d9", "#5cb85c", "#f0ad4e", "#d9534f", "#9b59b6", "#7f8c8d", "#2c3e50")


def box(ax, x, y, w, h, text, fc=C_BLUE, fs=10, tc="white", alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="white", lw=1.2, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=C_GRAY, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, color=color, lw=lw, linestyle=ls))


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已生成", name)


# ---------- 图1：Skill 三层架构 ----------
def fig1():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(6, 8.6, "Agent Skill 三层架构 — 文档质量评估技能", ha="center", fontsize=15, weight="bold")

    box(ax, 0.5, 7, 3, 0.9, "用户 / 智能体\n（提出质检需求）", C_GRAY, 10)
    box(ax, 8.5, 7, 3, 0.9, "触发关键词\n“文档质检/入库前检查”", C_GRAY, 9)

    layers = [
        (5.4, "元数据层（始终加载 · Token 极低）", "name + description\n告诉模型：这是什么技能", C_BLUE),
        (3.5, "指令层（触发时加载 · Token 中等）", "SKILL.md\n告诉模型：五大功能 SOP / 执行顺序 / 约束", C_GREEN),
        (1.0, "资源层（按需加载 · Token 较大）", "Python 模块 + assessment_config.yaml\nHTML 模板 / 单元测试", C_ORANGE),
    ]
    for y, title, body, c in layers:
        box(ax, 1.5, y + 0.55, 9, 0.75, title, c, 12)
        box(ax, 1.5, y - 0.55, 9, 0.95, body, "#ecf0f1", 10, tc=C_DARK)
    for y in (6.1, 4.2, 2.1):
        arrow(ax, 6, y + 0.35, 6, y + 0.15, C_DARK)
    arrow(ax, 3.5, 7.45, 4.5, 6.8); arrow(ax, 8.5, 7.45, 7.5, 6.8)

    box(ax, 3.5, 0.1, 5, 0.7, "输出：结构化 JSON 报告 + HTML 简报", C_PURPLE, 11)
    arrow(ax, 6, 0.5, 6, 0.85, C_PURPLE)
    save(fig, "01-Skill三层架构图.png")


# ---------- 图2：五大功能 ----------
def fig2():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(6, 7.6, "文档质量评估 Skill — 五大核心功能", ha="center", fontsize=15, weight="bold")
    box(ax, 0.3, 5.6, 2.6, 1, "输入\n文件夹 / 文件列表", C_GRAY, 11)

    mods = [
        ("① 格式分布统计", ".pdf/.docx/.md\n数量与占比", C_BLUE),
        ("② PDF 类型分流", "文字型/扫描型\n/混合型", C_GREEN),
        ("③ 长度分布", "P25~P99\n区间分布", C_ORANGE),
        ("④ 重复检测", "MD5 精确\nSimHash 近似", C_PURPLE),
        ("⑤ 敏感信息", "手机/邮箱\n身份证", C_RED),
    ]
    for i, (t, b, c) in enumerate(mods):
        x = 3.3 + i * 1.75
        box(ax, x, 5.4, 1.6, 1.2, t, c, 9)
        box(ax, x, 4.1, 1.6, 1.0, b, "#ecf0f1", 8.5, tc=C_DARK)
        arrow(ax, 2.9, 6.1, x, 6.0, C_GRAY, lw=1)

    box(ax, 3.3, 2.4, 8.6, 1, "汇总：分类标签 + 解析路由建议 + 三类待办列表", C_DARK, 12)
    for i in range(5):
        arrow(ax, 4.1 + i * 1.75, 4.1, 5.5, 3.4, C_GRAY, lw=0.8)

    box(ax, 1, 0.8, 4.5, 1, "结构化 JSON（机器可读）", C_BLUE, 11)
    box(ax, 6.5, 0.8, 4.5, 1, "HTML 简报（人工可读）", C_GREEN, 11)
    arrow(ax, 5, 2.4, 3.5, 1.8); arrow(ax, 7, 2.4, 8.5, 1.8)
    save(fig, "02-五大功能模块图.png")


# ---------- 图3：PDF 分流流程 ----------
def fig3():
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")
    ax.text(5, 12.6, "PDF 文字型/扫描型启发式分流流程", ha="center", fontsize=15, weight="bold")

    box(ax, 3.2, 11.3, 3.6, 0.8, "打开 PDF（容错）", C_GRAY, 11)
    box(ax, 3.2, 10, 3.6, 0.8, "逐页提取文本", C_BLUE, 11)
    box(ax, 2.6, 8.6, 4.8, 0.9, "该页非空白字符数\n< 阈值 100 ？", C_ORANGE, 10)
    arrow(ax, 5, 11.3, 5, 10.8); arrow(ax, 5, 10, 5, 9.5)

    box(ax, 0.4, 7.1, 3.4, 0.9, "是 → 扫描页 +1", C_RED, 10)
    box(ax, 6.2, 7.1, 3.4, 0.9, "否 → 文字页 +1", C_GREEN, 10)
    arrow(ax, 4.2, 8.6, 2.3, 8.0); arrow(ax, 5.8, 8.6, 7.7, 8.0)

    box(ax, 2.5, 5.7, 5, 0.9, "计算扫描页占比 ratio", C_BLUE, 11)
    arrow(ax, 2.1, 7.1, 4, 6.6); arrow(ax, 7.9, 7.1, 6, 6.6)

    box(ax, 0.3, 3.9, 2.8, 1.1, "ratio > 70%\nScan_PDF\n→ OCRParser", C_RED, 9.5)
    box(ax, 3.6, 3.9, 2.8, 1.1, "10%~70%\nHybrid_PDF\n→ HybridParser", C_ORANGE, 9.5)
    box(ax, 6.9, 3.9, 2.8, 1.1, "ratio ≤ 10%\nText_PDF\n→ DirectParser", C_GREEN, 9.5)
    arrow(ax, 4, 5.7, 1.8, 5.0); arrow(ax, 5, 5.7, 5, 5.0); arrow(ax, 6, 5.7, 8.2, 5.0)

    box(ax, 1.5, 2, 7, 1, "临界样本 ratio 60%~80% → 进入“待确认列表”", C_PURPLE, 11)
    box(ax, 1.5, 0.5, 7, 1, "打不开 / 0 页 → Corrupt_PDF → ReviewNode", C_DARK, 11)
    save(fig, "03-PDF分流流程图.png")


# ---------- 图4：分类标签体系 ----------
def fig4():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(6, 8.6, "文档分类标签体系（三维标签 + 路由）", ha="center", fontsize=15, weight="bold")

    dims = [
        (0.4, "格式维度", C_BLUE, ["Text_PDF", "Scan_PDF", "Hybrid_PDF", "Corrupt_PDF", "DOCX/MD/TXT"]),
        (4.35, "长度维度", C_GREEN, ["Empty_Doc", "Short_Doc", "Long_Doc", "（P25/P90 动态阈值）"]),
        (8.3, "风险维度", C_RED, ["Exact_Duplicate", "Near_Duplicate", "Sensitive_Info", "MD5/SimHash/正则"]),
    ]
    for x, title, c, tags in dims:
        box(ax, x, 6.8, 3.3, 0.8, title, c, 12)
        for i, t in enumerate(tags):
            box(ax, x, 5.7 - i * 0.95, 3.3, 0.75, t, "#ecf0f1", 9.5, tc=C_DARK)

    box(ax, 3, 1, 6, 0.9, "route_for()：按标签优先级决定唯一路由", C_PURPLE, 12)
    for x in (2, 6, 10):
        arrow(ax, x, 2.2, 6, 1.9, C_GRAY, lw=0.9)
    ax.text(6, 0.55, "优先级：损坏审核 > 安全审核 > 版本审核 > 去重 > OCR > 混合 > 直接解析",
            ha="center", fontsize=9.5, color=C_DARK)
    save(fig, "04-分类标签体系图.png")


# ---------- 图5：智能体工作流 ----------
def fig5():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(6, 8.6, "document_ingestion_workflow 智能体工作流", ha="center", fontsize=15, weight="bold")

    box(ax, 0.3, 6.6, 2.1, 0.9, "TriggerNode\n触发入库", C_GRAY, 9.5)
    box(ax, 3, 6.6, 3, 0.9, "DocumentQuality\nAssessmentSkill", C_BLUE, 9.5)
    box(ax, 6.7, 6.6, 2.1, 0.9, "DecisionNode\n标签路由", C_PURPLE, 9.5)
    arrow(ax, 2.4, 7.05, 3, 7.05); arrow(ax, 6, 7.05, 6.7, 7.05)

    nodes = [
        ("DirectParser", C_GREEN), ("OCRParser", C_RED), ("HybridParser", C_ORANGE),
        ("DedupeNode", C_GRAY), ("SecurityReview", "#c0392b"),
        ("VersionReview", "#d68910"), ("ReviewNode", C_DARK),
    ]
    for i, (n, c) in enumerate(nodes):
        x = 0.3 + i * 1.68
        box(ax, x, 4.3, 1.5, 0.9, n, c, 8.5)
        arrow(ax, 7.7, 6.6, x + 0.75, 5.2, C_GRAY, lw=0.8)

    box(ax, 1, 2.4, 10, 0.9,
        "示例：scan.pdf（Scan_PDF）→ 自动跳过直接解析 → OCRParser：渲染→旋转矫正→OCR→版面还原→入库",
        C_RED, 10)
    box(ax, 1, 1, 10, 0.9,
        "全程留痕：每个节点动作写入 trace，支持回放与审计；待审节点挂起不自动放行",
        C_DARK, 10)
    save(fig, "05-智能体工作流路由图.png")


for f in (fig1, fig2, fig3, fig4, fig5):
    f()
print("全部完成")
