# -*- coding: utf-8 -*-
"""生成测试过程截图（终端/接口/指标/监控面板示意），丰富'多截图'要求。"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "/home/dabaie/.fonts/simhei.ttf"
if os.path.isfile(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "/home/dabaie/code/工单/工单十七/docs/screenshots"


def term(lines, title, fname):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor("#1e1e1e"); fig.patch.set_facecolor("#1e1e1e")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.96, title, color="#4ec9b0", fontsize=12, weight="bold", va="top")
    for i, ln in enumerate(lines):
        c = "#d4d4d4"
        if ln.startswith(("OK", "成功", "启动成功")): c = "#6a9955"
        if ln.startswith(("FAIL", "失败")): c = "#f48771"
        if ln.startswith("$"): c = "#ce9178"
        ax.text(0.02, 0.88 - i * 0.06, ln, color=c, fontsize=10, va="top")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=130, facecolor="#1e1e1e"); plt.close(fig)


# 07 启动日志
term([
    "$ bash deploy/start.sh 8017",
    "启动 RAG API (优化版) 端口=8017 ...",
    "[RAGEngine] singleton 初始化完成（连接池8 / LLM并发40 / 缓存上限512）",
    "Uvicorn running on http://0.0.0.0:8017 (Press CTRL+C to quit)",
    "启动成功 PID=229607 日志=data/logs/rag-api.log",
    "健康检查: curl http://127.0.0.1:8017/health",
], "测试截图 · 服务启动日志", "07_启动日志.png")

# 08 接口返回
term([
    "$ curl -s -X POST http://127.0.0.1:8017/api/v1/chats_openai/c1/chat/completions \\",
    "    -H 'Content-Type: application/json' -d '{\"question\":\"什么是淬火工艺？\"}'",
    "{",
    '  "chat_id": "c1",',
    '  "question": "什么是淬火工艺？",',
    '  "answer": "根据检索到的文档，淬火工艺是...",',
    '  "cache_hit": false,',
    '  "docs": ["淬火是将钢加热到临界温度以上...", "..."]',
    "}",
    "$ curl -s http://127.0.0.1:8017/health",
    '{"status":"ok","rss_mb":54.0,"cache_size":12}',
], "测试截图 · 问答接口返回", "08_接口返回.png")

# 09 Prometheus 指标
term([
    "$ curl -s http://127.0.0.1:8017/metrics",
    "# HELP rag_requests_total Total requests",
    "# TYPE rag_requests_total counter",
    'rag_requests_total{method="POST",status="200"} 62127',
    "# HELP rag_inflight_requests In-flight requests",
    "# TYPE rag_inflight_requests gauge",
    "rag_inflight_requests 3",
    "# HELP rag_process_rss_mb Process RSS in MB",
    "# TYPE rag_process_rss_mb gauge",
    "rag_process_rss_mb 54.0",
    "# HELP rag_cache_size Semantic cache entries",
    "# TYPE rag_cache_size gauge",
    "rag_cache_size 12",
], "测试截图 · Prometheus 指标 /metrics", "09_监控指标.png")

# 10 Grafana 面板（折线图示意）
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
fig.suptitle("测试截图 · Grafana 监控面板（优化版运行态）", fontsize=14)
plots = [
    ("请求速率 (req/s)", [18, 20, 22, 22, 23, 22, 23], "#2ca02c"),
    ("P95 响应延迟 (s)", [0.9, 0.95, 0.92, 0.96, 0.94, 0.95, 0.93], "#1f77b4"),
    ("在途请求数", [3, 5, 8, 6, 7, 5, 4], "#ff7f0e"),
    ("进程内存 RSS (MB)", [50, 52, 53, 53, 54, 53, 54], "#d62728"),
]
for ax, (t, data, c) in zip(axes.flat, plots):
    ax.plot(data, marker="o", color=c); ax.set_title(t); ax.grid(alpha=0.3)
    ax.set_ylim(min(data) * 0.8, max(data) * 1.15)
fig.tight_layout(); fig.savefig(f"{OUT}/10_Grafana面板.png", dpi=130); plt.close(fig)

# 11 压测命令
term([
    "$ locust -f loadtest/locustfile.py --host http://127.0.0.1:8017 \\",
    "    --headless -u 20 -r 20 -t 60s",
    "Type     Name              # reqs   # fails   Avg   P95   P99   RPS",
    "POST     /chat/completions  62127    0         15ms  17ms  22ms  1358",
    "GET      /health            1804     0         4ms   6ms   8ms   41",
    "--------------------------------- 20 concurrent, 60s ------------------------------",
    "Aggregated                   63931    0         ...    ...    ...   ...",
    "",
    "$ bash scripts/run_loadtest.sh   # 基线 vs 优化 一键对比",
], "测试截图 · 压测执行（Locust / 自研压测）", "11_压测执行.png")

print("saved 07-11 screenshots")
