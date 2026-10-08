# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-RAG性能瓶颈识别与优化
src/perf_v13.py —— 工单十三 分阶段性能打点与结构化日志

任务要求：在每个主要处理阶段的进入/退出处实施带时间戳的结构化日志记录，
包含请求 ID 以追踪单个请求在流程中的执行情况，记录检索文档数、上下文长度、
生成 token 数等指标（工单十三：性能瓶颈识别）。

设计：
  - perf_v13.new_request(query)：生成请求 ID，写入线程上下文
  - perf_v13.stage(name, **extra)：上下文管理器，退出时记录该阶段耗时
  - 日志落盘 logs/perf_v13.jsonl（JSONL，一行一条），同时经 loguru 输出
  - 工作线程（检索线程池）通过 with_request_id() 传递请求 ID
"""
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from loguru import logger

WORK_ORDER = "人工智能NLP-RAG-RAG性能瓶颈识别与优化"

LOG_PATH = Path("logs/perf_v13.jsonl")
_ctx = threading.local()
_write_lock = threading.Lock()


def new_request(query: str = "") -> str:
    """工单十三：开始一次请求，生成本次请求 ID（主线程调用）"""
    rid = uuid.uuid4().hex[:8]
    _ctx.request_id = rid
    _ctx.query = (query or "")[:60]
    return rid


def current_request_id() -> str:
    return getattr(_ctx, "request_id", "-")


def with_request_id(rid: str, fn):
    """工单十三：包装函数——在工作线程中恢复请求 ID（线程池执行检索通道时使用）

    用法：pool.submit(with_request_id(rid, func), *args)
    """
    def wrapper(*args, **kwargs):
        old = getattr(_ctx, "request_id", None)
        _ctx.request_id = rid
        try:
            return fn(*args, **kwargs)
        finally:
            _ctx.request_id = old
    return wrapper


class _StageTimer:
    """工单十三：阶段计时器（暴露 .ms 供调用方读取）"""
    ms = 0.0


@contextmanager
def stage(name: str, **extra: Any):
    """工单十三：阶段计时上下文管理器，退出时写结构化日志

    用法：with stage("rerank", candidates=48) as s: ...; s.ms
    """
    timer = _StageTimer()
    t0 = time.perf_counter()
    try:
        yield timer
    finally:
        timer.ms = (time.perf_counter() - t0) * 1000
        record(name, timer.ms, **extra)


def record(name: str, ms: float, **extra: Any) -> None:
    """工单十三：写入一条阶段耗时记录（JSONL 结构化日志）"""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "request_id": current_request_id(),
           "query": getattr(_ctx, "query", ""),
           "stage": name, "ms": round(ms, 1)}
    rec.update(extra)
    logger.info(f"[perf_v13] {name}={rec['ms']}ms rid={rec['request_id']} "
                + " ".join(f"{k}={v}" for k, v in extra.items()))
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def clear_log() -> None:
    """工单十三：清空性能日志（每轮基准测试前调用）"""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            open(LOG_PATH, "w", encoding="utf-8").close()
    except Exception:
        pass
