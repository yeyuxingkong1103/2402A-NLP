# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""生成工单六设计文档所需的 5 张 PNG 图。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\作业\6-专高NLP 作业\成品\6\设计"
import os
os.makedirs(OUT, exist_ok=True)

C_MAIN = "#2563eb"; C_VEC = "#3b82f6"; C_TXT = "#f59e0b"; C_FUSE = "#8b5cf6"
C_RERANK = "#10b981"; C_LLM = "#ef4444"; C_BG = "#f8fafc"; C_DARK = "#1e293b"


def box(ax, x, y, w, h, text, fc=C_MAIN, fs=11, tc="white", ec="none"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=1.5, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold", zorder=3)


def arrow(ax, x1, y1, x2, y2, color=C_DARK, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=1.8, zorder=1))


def new_ax(w=14, h=9):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=C_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    return fig, ax


# ============ 1. 技术架构图 ============
def arch():
    fig, ax = new_ax()
    ax.text(5, 9.6, "混合检索 RAG 系统 · 技术架构图", ha="center", fontsize=18, weight="bold", color=C_DARK)

    box(ax, 0.3, 7.8, 2.2, 1.0, "Web 前端\n(策略配置面板)", "#64748b", 11)
    box(ax, 3.0, 7.8, 2.2, 1.0, "FastAPI\n/api/chat /search\n/retrieve_config", C_MAIN, 10)
    box(ax, 5.7, 7.8, 1.8, 1.0, "会话管理\n+Query改写", "#0ea5e9", 10)
    box(ax, 7.9, 7.8, 1.8, 1.0, "DeepSeek\nLLM 生成", C_LLM, 10)

    # 检索核心
    box(ax, 0.3, 5.6, 2.6, 1.1, "向量检索\nbge-m3 + Milvus\n余弦相似度", C_VEC, 11)
    box(ax, 3.7, 5.6, 2.6, 1.1, "全文检索\n倒排索引 BM25\n多字段加权", C_TXT, 11)
    box(ax, 7.1, 5.6, 2.6, 1.1, "融合算法\nRRF/加权/投票", C_FUSE, 11)

    box(ax, 2.4, 3.9, 5.2, 0.9, "重排器：CrossEncoder / TF-IDF / LLM", C_RERANK, 12)
    box(ax, 3.4, 2.3, 3.2, 0.9, "LRU 检索结果缓存", "#94a3b8", 11)
    box(ax, 0.3, 0.7, 3.0, 0.9, "Milvus 向量库\nrag_pdf_qa_v4", "#334155", 10)
    box(ax, 4.0, 0.7, 3.0, 0.9, "PDF解析\n文本/表格/图像语义", "#475569", 10)
    box(ax, 7.7, 0.7, 2.0, 0.9, "招股书1+2\n3448块", "#64748b", 10)

    arrow(ax, 2.5, 8.3, 3.0, 8.3)
    arrow(ax, 5.2, 8.3, 5.7, 8.3)
    arrow(ax, 7.5, 8.3, 7.9, 8.3)
    arrow(ax, 4.1, 7.8, 1.6, 6.7)
    arrow(ax, 4.1, 7.8, 5.0, 6.7)
    arrow(ax, 2.9, 5.6, 5.4, 5.05)
    arrow(ax, 7.6, 5.6, 6.0, 5.05)
    arrow(ax, 5.0, 3.9, 5.0, 3.2)
    arrow(ax, 1.6, 5.6, 1.8, 1.6)
    arrow(ax, 5.0, 2.3, 5.0, 1.6)
    fig.savefig(f"{OUT}\\01-技术架构图.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ============ 2. 混合检索流程图 ============
def flow():
    fig, ax = new_ax()
    ax.text(5, 9.6, "混合检索 · 检索流程图", ha="center", fontsize=18, weight="bold", color=C_DARK)

    box(ax, 3.8, 8.5, 2.4, 0.8, "用户 Query", C_DARK, 12)
    box(ax, 3.8, 7.2, 2.4, 0.8, "Query 改写\n(多轮指代消解)", "#0ea5e9", 10)
    box(ax, 3.8, 5.9, 2.4, 0.8, "策略选择\nvector/fulltext/hybrid", C_MAIN, 10)

    box(ax, 0.6, 4.0, 2.6, 1.2, "向量召回\nbge-m3 嵌入\nMilvus Top-K", C_VEC, 10)
    box(ax, 6.8, 4.0, 2.6, 1.2, "全文召回\nBM25 倒排索引\n布尔/短语/模糊", C_TXT, 10)

    box(ax, 3.4, 4.2, 3.2, 0.8, "结果融合\nRRF / 加权平均 / Borda投票", C_FUSE, 10)
    box(ax, 3.4, 2.8, 3.2, 0.8, "重排\nCrossEncoder / TF-IDF / LLM", C_RERANK, 10)
    box(ax, 3.4, 1.4, 3.2, 0.8, "Top-K 结果 → LLM 生成答案", "#334155", 11)

    arrow(ax, 5.0, 8.5, 5.0, 8.0)
    arrow(ax, 5.0, 7.2, 5.0, 6.7)
    arrow(ax, 4.4, 5.9, 2.4, 5.2)
    arrow(ax, 5.6, 5.9, 7.6, 5.2)
    arrow(ax, 1.9, 4.0, 4.4, 4.55)
    arrow(ax, 8.1, 4.0, 5.6, 4.55)
    arrow(ax, 5.0, 4.2, 5.0, 3.6)
    arrow(ax, 5.0, 2.8, 5.0, 2.2)

    # 侧边注释
    box(ax, 0.3, 7.2, 2.2, 1.0, "缓存命中\n直接返回", "#94a3b8", 10)
    arrow(ax, 2.5, 7.7, 3.8, 7.65)
    fig.savefig(f"{OUT}\\02-混合检索流程图.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ============ 3. 思维导图 ============
def mindmap():
    fig, ax = new_ax(14, 10)
    ax.text(5, 9.7, "混合检索任务 · 功能思维导图", ha="center", fontsize=18, weight="bold", color=C_DARK)
    box(ax, 4.0, 4.6, 2.0, 0.9, "混合检索\n策略配置", C_MAIN, 13)

    branches = [
        (1.5, 7.8, "向量检索", C_VEC, ["bge-m3 嵌入", "Milvus 余弦召回", "Top-K 相似度"]),
        (6.5, 7.8, "全文检索", C_TXT, ["倒排索引 BM25", "多字段加权", "布尔/短语/模糊"]),
        (0.6, 4.6, "融合算法", C_FUSE, ["RRF 倒数排名", "加权平均", "Borda 投票"]),
        (7.5, 4.6, "重排算法", C_RERANK, ["CrossEncoder", "TF-IDF", "LLM 打分"]),
        (2.2, 1.4, "性能优化", "#64748b", ["LRU 缓存", "配置热更新", "策略可插拔"]),
        (6.0, 1.4, "验收指标", "#0ea5e9", ["准确率≥90%", "召回率≥95%", "响应≤3秒"]),
    ]
    for x, y, title, color, subs in branches:
        box(ax, x, y, 2.0, 0.7, title, color, 12)
        arrow(ax, 5.0, 4.6 if y > 4.6 else 5.5, x + 1.0, y if y > 4.6 else y + 0.7)
        for i, s in enumerate(subs):
            sy = y - 0.55 - i * 0.5
            box(ax, x + 0.15, sy - 0.35, 1.7, 0.4, s, "#e2e8f0", 9, C_DARK)
    fig.savefig(f"{OUT}\\03-功能思维导图.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ============ 4. 接口文档图 ============
def api_doc():
    fig, ax = new_ax(14, 10)
    ax.text(5, 9.7, "混合检索 · API 接口一览", ha="center", fontsize=18, weight="bold", color=C_DARK)
    rows = [
        ("GET", "/api/retrieve_config", "获取当前检索策略配置", C_MAIN),
        ("POST", "/api/retrieve_config", "动态更新策略/融合/重排/权重", C_RERANK),
        ("GET", "/api/search", "纯检索，支持 strategy/fusion/reranker 参数", C_VEC),
        ("POST", "/api/chat", "问答，支持多轮+检索策略参数", C_FUSE),
        ("POST", "/api/ingest", "PDF 解析入库+重建全文索引", C_TXT),
        ("POST", "/api/upload", "上传 PDF 到 data 目录", "#64748b"),
        ("GET", "/api/health", "健康检查+缓存+检索配置", "#0ea5e9"),
        ("GET", "/api/session/{sid}", "获取会话历史", "#94a3b8"),
        ("DELETE", "/api/session/{sid}", "清空会话历史", "#94a3b8"),
    ]
    y = 8.9
    for method, path, desc, color in rows:
        box(ax, 0.4, y, 1.1, 0.55, method, color, 11)
        box(ax, 1.6, y, 3.0, 0.55, path, "#e2e8f0", 11, C_DARK)
        box(ax, 4.8, y, 4.8, 0.55, desc, "#f1f5f9", 10, C_DARK)
        y -= 0.85
    fig.savefig(f"{OUT}\\04-接口文档.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ============ 5. 技术组件图 ============
def components():
    fig, ax = new_ax(14, 10)
    ax.text(5, 9.7, "混合检索 · 技术组件清单", ha="center", fontsize=18, weight="bold", color=C_DARK)
    groups = [
        (0.4, 6.2, "检索引擎", C_MAIN, [
            ("向量检索", "bge-m3 / Milvus / COSINE"),
            ("全文检索", "BM25 / jieba / rank_bm25"),
            ("融合算法", "RRF / Weighted / Voting"),
        ]),
        (5.2, 6.2, "重排组件", C_RERANK, [
            ("CrossEncoder", "bge-reranker-large"),
            ("TF-IDF", "轻量余弦相似度"),
            ("LLM 重排", "DeepSeek 批量打分"),
        ]),
        (0.4, 2.6, "数据层", C_TXT, [
            ("向量库", "Milvus rag_pdf_qa_v4"),
            ("倒排索引", "内存 BM25 多字段"),
            ("PDF 解析", "文本/表格/图像语义"),
        ]),
        (5.2, 2.6, "服务层", C_FUSE, [
            ("FastAPI", "异步 + SSE 流式"),
            ("缓存", "LRU 检索/LLM 缓存"),
            ("配置", "检索策略热更新"),
        ]),
    ]
    for x, y, title, color, items in groups:
        box(ax, x, y + 2.3, 4.4, 0.7, title, color, 13)
        for i, (k, v) in enumerate(items):
            iy = y + 1.5 - i * 0.75
            box(ax, x + 0.1, iy, 1.7, 0.55, k, "#e2e8f0", 10, C_DARK)
            box(ax, x + 1.9, iy, 2.4, 0.55, v, "#f1f5f9", 9, C_DARK)
    fig.savefig(f"{OUT}\\05-技术组件.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


if __name__ == "__main__":
    arch(); flow(); mindmap(); api_doc(); components()
    print("5 张设计图已生成：", OUT)
