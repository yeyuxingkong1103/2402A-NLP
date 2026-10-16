# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 评估框架集中配置。

本模块只放**常量与路径**，import 时不产生副作用（不读索引、不发网络请求、不改环境变量）；
需要环境变量的调用方显式调用 :func:`apply_env`（由 ``运行评估.py`` 在 import app 之前调用）。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 工单与路径
# ---------------------------------------------------------------------------
WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"

#: 仓库根（本文件位于 <root>/研发/评估/配置.py）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_DIR = PROJECT_ROOT / "研发" / "评估"
QUESTION_SET = EVAL_DIR / "测试问题集.jsonl"

#: 面向用户的交付目录（CSV / JSONL / 汇总 / 分析 / 日志都落这里）
OUT_DIR = PROJECT_ROOT / "测试" / "测试数据"

DATA_DIR = PROJECT_ROOT / "研发" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
LOG_DIR = PROJECT_ROOT / "部署" / "日志"

#: 索引清单（开工前的就绪判据：chunk_total 必须 ≥ INDEX_MIN_CHUNKS）
INDEX_MANIFEST = INDEX_DIR / "index_manifest.json"
INDEX_MIN_CHUNKS = 8000

# ---------------------------------------------------------------------------
# 环境变量（供 apply_env() 使用；已存在的环境变量不覆盖）
# ---------------------------------------------------------------------------
ENV_DEFAULTS = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": str(PROJECT_ROOT / "研发"),
    "RAG_DATA__RAW_DIR": str(RAW_DIR),
    "RAG_DATA__PROCESSED_DIR": str(PROCESSED_DIR),
    "RAG_DATA__INDEX_DIR": str(INDEX_DIR),
    "RAG_LOG__DIR": str(LOG_DIR),
}

APP_DIR = PROJECT_ROOT / "研发"

# ---------------------------------------------------------------------------
# 检索 / 生成参数
# ---------------------------------------------------------------------------
TOP_K = 10                 # 评估统一 top_k
PROMPT_TOP_K = 3           # 生成层实际取用的证据条数（引擎配置 retrieval.prompt_top_k）
RECALL_KS = (5, 10)        # 需要同时报告的 recall@k

#: 关闭答案缓存：每题都是"用户第一次问"，才能量到真实的检索+生成耗时。
#: （引擎默认 use_cache=True，第二次问同一题会命中落盘缓存 → total≈0ms，会污染 SLA 统计。）
USE_CACHE = False

SESSION_PREFIX = "eval7"

# ---------------------------------------------------------------------------
# 判分阈值
# ---------------------------------------------------------------------------
#: 关键词覆盖率 ≥ 该值且未拒答 ⇒ 判对（严格口径）
KEYWORD_THRESHOLD = 0.6
#: 覆盖率落在 [PARTIAL_THRESHOLD, KEYWORD_THRESHOLD) ⇒ 判为「部分正确」（另见 判分.partial_correct）
PARTIAL_THRESHOLD = 0.3
#: 数值原子覆盖率 ≥ 该值也可支撑「部分正确」（用于中英混排/单位换算导致的表述差异）
ATOMIC_THRESHOLD = 0.6

#: 响应时间 SLA
SLA_MS = 3000

#: 跨语料泄漏只看前 N 条命中
LEAK_TOP_N = 3

# ---------------------------------------------------------------------------
# 语料（9 份 A 股年报；由 测试问题集.jsonl 的 source_file 字段派生，仅作校验用）
# ---------------------------------------------------------------------------
EXPECTED_CORPUS_COUNT = 9

OUT_FILES = {
    "jsonl": OUT_DIR / "评估结果.jsonl",
    "csv": OUT_DIR / "评估结果.csv",
    "summary": OUT_DIR / "评估汇总.json",
    "analysis": OUT_DIR / "检索问题分析.md",
    "log": OUT_DIR / "评估运行日志.txt",
}


def apply_env(override: bool = False) -> None:
    """把 :data:`ENV_DEFAULTS` 灌进 ``os.environ``（默认不覆盖已有值）。"""
    for key, value in ENV_DEFAULTS.items():
        if override or not os.environ.get(key):
            os.environ[key] = value


def ensure_out_dir() -> Path:
    """确保输出目录存在（含父目录），返回该目录。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
