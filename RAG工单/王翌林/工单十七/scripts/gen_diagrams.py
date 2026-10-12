# -*- coding: utf-8 -*-
"""生成架构流程图与思维导图 PNG。"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

FONT = "/home/dabaie/.fonts/simhei.ttf"
if os.path.isfile(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/home/dabaie/code/工单/工单十七/docs/screenshots"


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color="#333", lw=1.2))


# ---------- 流程图 ----------
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 3.8, 9.0, 2.4, 0.7, "客户端请求", "#cfe8ff")
box(ax, 3.8, 7.8, 2.4, 0.7, "FastAPI 中间件\n(指标/在途计数)", "#cfe8ff")
box(ax, 1.5, 6.4, 2.6, 0.7, "语义缓存命中?", "#fff2cc")
box(ax, 5.9, 6.4, 2.6, 0.7, "文档解析请求", "#fff2cc")
box(ax, 1.5, 5.0, 2.6, 0.7, "连接池借连接\n向量检索 TopK", "#e2efda")
box(ax, 5.9, 5.0, 2.6, 0.7, "入 asyncio.Queue\n返回 task_id", "#e2efda")
box(ax, 1.5, 3.6, 2.6, 0.7, "归还连接(finally)", "#e2efda")
box(ax, 5.9, 3.6, 2.6, 0.7, "消费者异步\nDeepDoc/VLM 解析", "#e2efda")
box(ax, 1.5, 2.2, 2.6, 0.7, "LLM 并发限流\nSemaphore(40)", "#fce4d6")
box(ax, 1.5, 0.8, 2.6, 0.7, "生成→释放中间对象\n写缓存→返回", "#fce4d6")
box(ax, 6.6, 1.5, 2.6, 0.7, "Prometheus /metrics", "#d9d9d9")

arrow(ax, 5, 9.0, 5, 8.5); arrow(ax, 5, 7.8, 3.0, 7.1); arrow(ax, 5, 7.8, 7.2, 7.1)
arrow(ax, 2.8, 6.4, 2.8, 5.7); arrow(ax, 7.2, 6.4, 7.2, 5.7)
arrow(ax, 2.8, 5.0, 2.8, 4.3); arrow(ax, 2.8, 3.6, 2.8, 2.9); arrow(ax, 2.8, 2.2, 2.8, 1.5)
arrow(ax, 7.2, 5.0, 7.2, 4.3)
ax.text(2.3, 6.75, "否", fontsize=9, color="green"); ax.text(0.3, 6.75, "是→返回", fontsize=9, color="gray")
ax.set_title("工单十七 · 系统流程图", fontsize=14)
fig.tight_layout(); fig.savefig(f"{OUT}/05_流程图.png", dpi=140); plt.close(fig)

# ---------- 思维导图 ----------
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
box(ax, 4.5, 3.4, 3, 1.2, "并发瓶颈与\n资源泄漏治理", "#cfe8ff")
topics = [
    (0.3, 6.6, "设计", ["技术组件", "技术架构", "流程图/思维导图", "接口文档"]),
    (0.3, 3.9, "研发", ["RAGEngine 单例", "连接池/LRU缓存", "异步队列/限流", "资源释放"]),
    (0.3, 1.2, "测试", ["Locust/JMeter", "20并发场景A", "内存曲线", "多截图对比"]),
    (8.8, 6.6, "优化", ["消除每请求初始化", "有界资源防泄漏", "LLM并发限流", "显式gc/empty_cache"]),
    (8.8, 3.9, "部署", ["install.sh conda+cuda", "start.sh 启动", "stop.sh 结束", "docker-compose"]),
    (8.8, 1.2, "监控", ["Prometheus", "Grafana面板", "RSS/延迟告警", "12h长稳"]),
]
for cx, cy, title, subs in topics:
    fc = "#fce4d6" if title in ("设计", "优化", "部署") else "#e2efda"
    box(ax, cx, cy, 2.6, 0.7, title, fc)
    arrow(ax, 6 if cx < 6 else 6, 4, cx + 1.3, cy + 0.35)
    for i, s in enumerate(subs):
        ax.text(cx + 0.1, cy - 0.35 - i * 0.32, "• " + s, fontsize=9, va="center")
ax.set_title("工单十七 · 思维导图", fontsize=14)
fig.tight_layout(); fig.savefig(f"{OUT}/06_思维导图.png", dpi=140); plt.close(fig)
print("saved 05_流程图.png 06_思维导图.png")
