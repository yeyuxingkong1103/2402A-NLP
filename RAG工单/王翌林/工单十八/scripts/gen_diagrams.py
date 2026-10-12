# -*- coding: utf-8 -*-
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

FONT = "/home/dabaie/.fonts/simhei.ttf"
if os.path.isfile(FONT):
    font_manager.fontManager.addfont(FONT); plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "/home/dabaie/code/工单/工单十八/docs/screenshots"


def box(ax, x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#333"))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9.5)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color="#333"))


fig, ax = plt.subplots(figsize=(11, 6.5)); ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")
box(ax, 4, 6.0, 3, 0.7, "触发：入库质检", "#cfe8ff")
box(ax, 4, 4.9, 3, 0.7, "DocumentQualityAssessmentSkill", "#fff2cc")
steps = [("格式分布统计", 0.3), ("PDF 文字/扫描/混合 分流", 2.3),
         ("长度分布 P25~P99", 4.3), ("MD5+SimHash 去重", 6.3), ("敏感信息检测", 8.3)]
for t, x in steps:
    box(ax, x, 3.4, 1.9, 0.7, t, "#e2efda")
box(ax, 2.5, 1.9, 3, 0.7, "分类打标签 + 路由 route_map", "#fce4d6")
box(ax, 6, 1.9, 3, 0.7, "输出 JSON + HTML 简报", "#fce4d6")
box(ax, 1.5, 0.5, 2.2, 0.7, "Text_PDF→文本提取", "#d9d9d9")
box(ax, 4.4, 0.5, 2.2, 0.7, "Scan_PDF→OCR 解析", "#d9d9d9")
box(ax, 7.3, 0.5, 2.2, 0.7, "待人工复核 列表", "#d9d9d9")
arrow(ax, 5.5, 6.0, 5.5, 5.6)
for x in [1.25, 3.25, 5.25, 7.25, 9.25]:
    arrow(ax, 5.5, 4.9, x, 4.1)
    arrow(ax, x, 3.4, 4, 2.6)
arrow(ax, 5.5, 3.4, 7.5, 2.6)
arrow(ax, 3.2, 1.9, 2.6, 1.2); arrow(ax, 4.8, 1.9, 5.5, 1.2); arrow(ax, 6.5, 1.9, 8.4, 1.2)
ax.set_title("工单十八 · 质检 Skill 与工作流流程图", fontsize=14)
fig.tight_layout(); fig.savefig(f"{OUT}/10_流程图.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
box(ax, 4.3, 2.9, 3.4, 1.2, "文档质量评估 Skill\n集成至智能体工作流", "#cfe8ff")
topics = [
    (0.2, 5.6, "设计", ["SKILL.md 三层结构", "assessment_config.yaml", "流程图/思维导图", "接口文档"]),
    (0.2, 2.9, "研发(五大功能)", ["格式分布统计", "PDF 文字/扫描分流", "长度分位分布", "MD5+SimHash 去重", "敏感信息检测"]),
    (0.2, 0.3, "测试", ["单元测试", "模拟数据集验证", "API 冒烟测试", "多截图对比"]),
    (8.6, 5.6, "集成", ["document_ingestion_workflow", "分类标签路由 route_map", "Scan_PDF→OCR", "POST 质检 API"]),
    (8.6, 2.9, "优化", ["阈值全配置化", "错误处理(损坏PDF)", "启发式→待确认列表", "进度反馈/可中断"]),
    (8.6, 0.3, "部署", ["install.sh conda", "start.sh / stop.sh", "Docker 可选", "演示视频脚本"]),
]
for cx, cy, title, subs in topics:
    fc = "#fce4d6" if title in ("设计", "优化", "部署") else "#e2efda"
    box(ax, cx, cy, 2.8, 0.7, title, fc)
    arrow(ax, 6 if cx < 6 else 6, 3.5, cx + 1.4, cy + 0.35)
    for i, s in enumerate(subs):
        ax.text(cx + 0.1, cy - 0.35 - i * 0.3, "- " + s, fontsize=8.5, va="center")
ax.set_title("工单十八 · 思维导图", fontsize=14)
fig.tight_layout(); fig.savefig(f"{OUT}/11_思维导图.png", dpi=140); plt.close(fig)
print("saved 10_流程图.png 11_思维导图.png")
