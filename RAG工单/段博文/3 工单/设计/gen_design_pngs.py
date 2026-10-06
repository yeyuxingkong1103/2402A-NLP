# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""生成工单三设计图：技术架构图 + 思维导图 + 流程图。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = r"D:\作业\6-专高NLP 作业\成品\3\设计"

# ========== 图1：技术架构图 ==========
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("RAG-PDF 问答系统技术架构图（表格解析增强版）", fontsize=18, weight="bold", pad=20)

# 定义层级
layers = [
    {"y": 8.5, "label": "用户层", "color": "#dbeafe", "items": ["Web 界面", "API 接口"]},
    {"y": 6.5, "label": "服务层", "color": "#dcfce7", "items": ["FastAPI", "上传/入库", "检索/问答", "缓存管理"]},
    {"y": 4.5, "label": "处理层", "color": "#fef3c7", "items": ["PDF 解析\n(PyMuPDF)", "表格提取\n(pdfplumber)", "文本切分", "向量化\n(bge-m3)"]},
    {"y": 2.5, "label": "检索层", "color": "#fce7f3", "items": ["向量检索\n(Milvus)", "BM25 检索\n(jieba)", "RRF 融合", "重排\n(bge-reranker)"]},
    {"y": 0.5, "label": "生成层", "color": "#e0e7ff", "items": ["LLM 生成\n(DeepSeek)", "流式输出", "答案缓存"]},
]

box_width = 2.2
box_height = 0.9
spacing = 0.3

for layer in layers:
    y = layer["y"]
    label = layer["label"]
    color = layer["color"]
    items = layer["items"]

    # 层级标签
    ax.text(0.5, y + box_height/2, label, fontsize=14, weight="bold", va="center")

    # 计算起始 x 使居中
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
        ax.text(x + box_width/2, y + box_height/2, item, fontsize=9, ha="center", va="center")

# 画箭头
for i in range(len(layers) - 1):
    y1 = layers[i]["y"]
    y2 = layers[i+1]["y"] + box_height
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
ax.set_title("RAG-PDF 问答系统思维导图（表格解析增强版）", fontsize=18, weight="bold", pad=20)

# 中心节点
center_x, center_y = 7, 5
center = mpatches.FancyBboxPatch(
    (center_x - 1.2, center_y - 0.5), 2.4, 1.0,
    boxstyle="round,pad=0.1",
    facecolor="#3b82f6", edgecolor="#1e40af", linewidth=2
)
ax.add_patch(center)
ax.text(center_x, center_y, "RAG-PDF\n问答系统", fontsize=12, weight="bold", color="white", ha="center", va="center")

# 分支
branches = [
    {"pos": (3, 8), "label": "数据层", "color": "#dbeafe", "items": ["PDF 上传", "表格解析", "文本切分", "向量化"]},
    {"pos": (11, 8), "label": "检索层", "color": "#dcfce7", "items": ["向量检索", "BM25 检索", "RRF 融合", "重排"]},
    {"pos": (3, 2), "label": "生成层", "color": "#fef3c7", "items": ["LLM 调用", "流式输出", "缓存优化", "错误处理"]},
    {"pos": (11, 2), "label": "优化层", "color": "#fce7f3", "items": ["表格识别", "结构保留", "语义增强", "性能优化"]},
]

for branch in branches:
    bx, by = branch["pos"]
    label = branch["label"]
    color = branch["color"]
    items = branch["items"]

    # 分支框
    rect = mpatches.FancyBboxPatch(
        (bx - 1.5, by - 0.8), 3.0, 1.6 + len(items) * 0.4,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#64748b", linewidth=1.5
    )
    ax.add_patch(rect)

    # 标签
    ax.text(bx, by + 0.5, label, fontsize=11, weight="bold", ha="center", va="center")

    # 子项
    for i, item in enumerate(items):
        ax.text(bx, by - 0.1 - i * 0.4, f"• {item}", fontsize=9, ha="center", va="top")

    # 连线
    ax.plot([center_x, bx], [center_y, by], color="#64748b", linewidth=1.5, alpha=0.6)

fig.savefig(OUT_DIR + r"\02-思维导图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[生成] 02-思维导图.png")

# ========== 图3：表格解析流程图 ==========
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("表格解析流程图", fontsize=18, weight="bold", pad=20)

steps = [
    {"y": 8.5, "label": "PDF 文件输入", "color": "#dbeafe", "desc": "招股说明书1.pdf / 招股说明书2.pdf"},
    {"y": 7.0, "label": "PyMuPDF 文本提取", "color": "#dcfce7", "desc": "提取普通文本块"},
    {"y": 5.5, "label": "pdfplumber 表格识别", "color": "#fef3c7", "desc": "识别表格结构（行列）"},
    {"y": 4.0, "label": "表格转 Markdown", "color": "#fce7f3", "desc": "保留行列语义结构"},
    {"y": 2.5, "label": "文本切分", "color": "#e0e7ff", "desc": "400字符/50重叠，表格不切分"},
    {"y": 1.0, "label": "向量化入库", "color": "#d1fae5", "desc": "bge-m3 向量化 → Milvus"},
]

box_width = 4.0
box_height = 0.9

for i, step in enumerate(steps):
    y = step["y"]
    label = step["label"]
    color = step["color"]
    desc = step["desc"]

    # 框
    rect = mpatches.FancyBboxPatch(
        (7 - box_width/2, y), box_width, box_height,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#64748b", linewidth=1.5
    )
    ax.add_patch(rect)

    # 标签
    ax.text(7, y + box_height/2, label, fontsize=11, weight="bold", ha="center", va="center")

    # 描述
    ax.text(7, y - 0.25, desc, fontsize=9, ha="center", va="top", color="#64748b")

    # 箭头
    if i < len(steps) - 1:
        ax.annotate("", xy=(7, steps[i+1]["y"] + box_height), xytext=(7, y),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))

fig.savefig(OUT_DIR + r"\03-表格解析流程图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[生成] 03-表格解析流程图.png")

# ========== 图4：检索问答流程图 ==========
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("检索问答流程图", fontsize=18, weight="bold", pad=20)

steps = [
    {"y": 8.5, "label": "用户提问", "color": "#dbeafe", "desc": "接收用户问题"},
    {"y": 7.0, "label": "混合检索", "color": "#dcfce7", "desc": "向量检索 + BM25 检索"},
    {"y": 5.5, "label": "RRF 融合", "color": "#fef3c7", "desc": "融合两路检索结果"},
    {"y": 4.0, "label": "重排过滤", "color": "#fce7f3", "desc": "bge-reranker 重排，阈值过滤"},
    {"y": 2.5, "label": "LLM 生成", "color": "#e0e7ff", "desc": "DeepSeek API 生成答案"},
    {"y": 1.0, "label": "返回答案", "color": "#d1fae5", "desc": "返回答案 + 引用来源"},
]

for i, step in enumerate(steps):
    y = step["y"]
    label = step["label"]
    color = step["color"]
    desc = step["desc"]

    rect = mpatches.FancyBboxPatch(
        (7 - box_width/2, y), box_width, box_height,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#64748b", linewidth=1.5
    )
    ax.add_patch(rect)

    ax.text(7, y + box_height/2, label, fontsize=11, weight="bold", ha="center", va="center")
    ax.text(7, y - 0.25, desc, fontsize=9, ha="center", va="top", color="#64748b")

    if i < len(steps) - 1:
        ax.annotate("", xy=(7, steps[i+1]["y"] + box_height), xytext=(7, y),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))

fig.savefig(OUT_DIR + r"\04-检索问答流程图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[生成] 04-检索问答流程图.png")

# ========== 图5：优化对比图 ==========
fig, ax = plt.subplots(figsize=(12, 7))

categories = ["分块大小", "召回数量", "重排数量", "LLM tokens", "温度"]
before = [500, 10, 5, 2048, 0.3]
after = [400, 6, 3, 768, 0.1]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, before, width, label="优化前（工单一）", color="#fca5a5")
bars2 = ax.bar(x + width/2, after, width, label="优化后（工单三）", color="#86efac")

ax.set_xlabel("参数", fontsize=12)
ax.set_ylabel("数值", fontsize=12)
ax.set_title("优化前后参数对比", fontsize=14, weight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# 添加数值标签
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
