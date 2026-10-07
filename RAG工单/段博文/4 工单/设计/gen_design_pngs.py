# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""生成工单四设计图：技术架构图 + 思维导图 + 图像解析流程图 + 检索流程图 + 优化对比。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = r"D:\作业\6-专高NLP 作业\成品\4\设计"

# ========== 图1：技术架构图 ==========
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("RAG-PDF 问答系统技术架构图（图像内容解析增强版）", fontsize=18, weight="bold", pad=20)

layers = [
    {"y": 8.5, "label": "用户层", "color": "#dbeafe", "items": ["Web 界面", "API 接口"]},
    {"y": 6.5, "label": "服务层", "color": "#dcfce7", "items": ["FastAPI", "上传/入库", "检索/问答", "缓存管理"]},
    {"y": 4.5, "label": "处理层", "color": "#fef3c7", "items": ["PDF解析\n(PyMuPDF)", "表格提取\n(pdfplumber)", "图像语义\n(OCR+LLM)", "文本切分", "向量化\n(bge-m3)"]},
    {"y": 2.5, "label": "检索层", "color": "#fce7f3", "items": ["向量检索\n(Milvus)", "BM25检索\n(jieba)", "RRF 融合", "重排\n(bge-reranker)"]},
    {"y": 0.5, "label": "生成层", "color": "#e0e7ff", "items": ["LLM 生成\n(DeepSeek)", "流式输出", "答案缓存"]},
]

box_width = 1.9
box_height = 0.9
spacing = 0.28

for layer in layers:
    y = layer["y"]
    color = layer["color"]
    items = layer["items"]

    ax.text(0.5, y + box_height / 2, layer["label"], fontsize=14, weight="bold", va="center")

    total_width = len(items) * box_width + (len(items) - 1) * spacing
    start_x = (16 - total_width) / 2 + 1

    for i, item in enumerate(items):
        x = start_x + i * (box_width + spacing)
        rect = mpatches.FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#64748b", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + box_width / 2, y + box_height / 2, item, fontsize=8.5, ha="center", va="center")

for i in range(len(layers) - 1):
    y1 = layers[i]["y"]
    y2 = layers[i + 1]["y"] + box_height
    ax.annotate("", xy=(8, y2), xytext=(8, y1),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))

fig.savefig(OUT_DIR + r"\01-技术架构图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[生成] 01-技术架构图.png")

# ========== 图2：思维导图 ==========
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("RAG-PDF 问答系统思维导图（图像内容解析增强版）", fontsize=18, weight="bold", pad=20)

center_x, center_y = 7, 5
center = mpatches.FancyBboxPatch(
    (center_x - 1.2, center_y - 0.5), 2.4, 1.0,
    boxstyle="round,pad=0.1",
    facecolor="#3b82f6", edgecolor="#1e40af", linewidth=2
)
ax.add_patch(center)
ax.text(center_x, center_y, "RAG-PDF\n问答系统", fontsize=12, weight="bold", color="white", ha="center", va="center")

branches = [
    {"pos": (3, 8), "label": "数据层", "color": "#dbeafe", "items": ["PDF 上传", "文本/表格解析", "图像语义解析", "切分向量化"]},
    {"pos": (11, 8), "label": "检索层", "color": "#dcfce7", "items": ["向量检索", "BM25 检索", "RRF 融合", "重排"]},
    {"pos": (3, 2), "label": "生成层", "color": "#fef3c7", "items": ["LLM 调用", "流式输出", "缓存优化", "错误处理"]},
    {"pos": (11, 2), "label": "优化层", "color": "#fce7f3", "items": ["OCR 识别", "空间就近配对", "结构块保留", "噪声过滤"]},
]

for branch in branches:
    bx, by = branch["pos"]
    color = branch["color"]
    items = branch["items"]

    rect = mpatches.FancyBboxPatch(
        (bx - 1.5, by - 0.8), 3.0, 1.6 + len(items) * 0.4,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#64748b", linewidth=1.5
    )
    ax.add_patch(rect)

    ax.text(bx, by + 0.5, branch["label"], fontsize=11, weight="bold", ha="center", va="center")

    for i, item in enumerate(items):
        ax.text(bx, by - 0.1 - i * 0.4, f"• {item}", fontsize=9, ha="center", va="top")

    ax.plot([center_x, bx], [center_y, by], color="#64748b", linewidth=1.5, alpha=0.6)

fig.savefig(OUT_DIR + r"\02-思维导图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[生成] 02-思维导图.png")

# ========== 图3：图像内容解析流程图 ==========
fig, ax = plt.subplots(figsize=(14, 10.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10.5)
ax.axis("off")
ax.set_title("图像内容解析流程图", fontsize=18, weight="bold", pad=20)

steps = [
    {"y": 9.0, "label": "PDF 文件输入", "color": "#dbeafe", "desc": "招股说明书1.pdf / 招股说明书2.pdf"},
    {"y": 7.5, "label": "定位含图页", "color": "#dcfce7", "desc": "内嵌大图 / 图引用词 + 矢量笔画数达标"},
    {"y": 6.0, "label": "图像获取", "color": "#fef3c7", "desc": "内嵌位图逐张提取 / 纯矢量图整页 2x 高分辨率渲染"},
    {"y": 4.5, "label": "OCR 识别（RapidOCR）", "color": "#fce7f3", "desc": "识别文字并保留空间坐标，去水印"},
    {"y": 3.0, "label": "密度门槛 + 空间配对", "color": "#e0e7ff", "desc": "过滤照片/印章；标签与百分比按 y 就近配对"},
    {"y": 1.5, "label": "LLM 生成语义块并入库", "color": "#d1fae5", "desc": "DeepSeek 生成语义，block_type=image → Milvus"},
]

box_width = 4.4
box_height = 0.9

for i, step in enumerate(steps):
    y = step["y"]

    rect = mpatches.FancyBboxPatch(
        (7 - box_width / 2, y), box_width, box_height,
        boxstyle="round,pad=0.05",
        facecolor=step["color"], edgecolor="#64748b", linewidth=1.5
    )
    ax.add_patch(rect)

    ax.text(7, y + box_height / 2, step["label"], fontsize=11, weight="bold", ha="center", va="center")
    ax.text(7, y - 0.25, step["desc"], fontsize=9, ha="center", va="top", color="#475569")

    if i < len(steps) - 1:
        ax.annotate("", xy=(7, steps[i + 1]["y"] + box_height), xytext=(7, y),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))

fig.savefig(OUT_DIR + r"\03-图像解析流程图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[生成] 03-图像解析流程图.png")

# ========== 图4：检索问答流程图 ==========
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("检索问答流程图", fontsize=18, weight="bold", pad=20)

steps = [
    {"y": 8.5, "label": "用户提问", "color": "#dbeafe", "desc": "接收用户问题（含图像类问题）"},
    {"y": 7.0, "label": "混合检索", "color": "#dcfce7", "desc": "向量检索 + BM25 检索（含图像语义块）"},
    {"y": 5.5, "label": "RRF 融合", "color": "#fef3c7", "desc": "融合两路检索结果"},
    {"y": 4.0, "label": "重排过滤", "color": "#fce7f3", "desc": "bge-reranker 重排，阈值过滤，保留 Top3"},
    {"y": 2.5, "label": "LLM 生成", "color": "#e0e7ff", "desc": "DeepSeek 生成，强化表格/图像理解"},
    {"y": 1.0, "label": "返回答案", "color": "#d1fae5", "desc": "返回答案 + 引用来源（标注页码/块类型）"},
]

box_width = 4.4
for i, step in enumerate(steps):
    y = step["y"]

    rect = mpatches.FancyBboxPatch(
        (7 - box_width / 2, y), box_width, box_height,
        boxstyle="round,pad=0.05",
        facecolor=step["color"], edgecolor="#64748b", linewidth=1.5
    )
    ax.add_patch(rect)

    ax.text(7, y + box_height / 2, step["label"], fontsize=11, weight="bold", ha="center", va="center")
    ax.text(7, y - 0.25, step["desc"], fontsize=9, ha="center", va="top", color="#475569")

    if i < len(steps) - 1:
        ax.annotate("", xy=(7, steps[i + 1]["y"] + box_height), xytext=(7, y),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))

fig.savefig(OUT_DIR + r"\04-检索问答流程图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[生成] 04-检索问答流程图.png")

# ========== 图5：优化前后参数对比 ==========
fig, ax = plt.subplots(figsize=(12, 7))

categories = ["分块大小", "召回数量", "重排数量", "LLM tokens", "温度"]
before = [500, 10, 5, 2048, 0.3]
after = [400, 6, 3, 768, 0.1]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width / 2, before, width, label="优化前（工单一）", color="#fca5a5")
bars2 = ax.bar(x + width / 2, after, width, label="优化后（工单四）", color="#86efac")

ax.set_xlabel("参数", fontsize=12)
ax.set_ylabel("数值", fontsize=12)
ax.set_title("优化前后参数对比", fontsize=14, weight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:g}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

fig.savefig(OUT_DIR + r"\05-优化前后参数对比.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[生成] 05-优化前后参数对比.png")

print("\n全部设计图生成完成！")
