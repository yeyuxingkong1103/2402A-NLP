# -*- coding: utf-8 -*-
"""
V13 性能配置
工单编号: 人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === 基线模拟: 故意慢 ===
BASELINE_SLEEP_QUERY = 0.8       # Query 增强延迟 (秒)
BASELINE_SLEEP_RETRIEVE = 1.5    # 检索延迟
BASELINE_SLEEP_CONTEXT = 0.5     # 上下文组装延迟
BASELINE_SLEEP_LLM = 3.0         # LLM 生成延迟
BASELINE_SLEEP_POST = 0.3        # 后处理延迟

# === 优化后 ===
OPT_SLEEP_QUERY = 0.05
OPT_SLEEP_RETRIEVE = 0.3
OPT_SLEEP_CONTEXT = 0.1
OPT_SLEEP_LLM = 1.5
OPT_SLEEP_POST = 0.05

# === 开关 ===
USE_CACHE = True
USE_PARALLEL = True
USE_SMALL_MODEL = True           # 小模型代替大模型
USE_CONTEXT_PRUNING = True       # 上下文剪枝
USE_QUERY_SKIP = True            # 跳过 query rewrite
USE_ASYNC_POST = True            # 异步后处理

# === 缓存 ===
CACHE_TTL = 3600
CACHE_MAX_SIZE = 1000

# === Top-K ===
BASELINE_TOP_K = 10              # 基线大 Top-K
OPT_TOP_K = 5                    # 优化后剪枝

# === 验收阈值 ===
MAX_LATENCY_MS = 3000            # 3 秒
