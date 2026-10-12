# -*- coding: utf-8 -*-
"""
perf.py — RAG 性能分析工具
工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

提供性能瓶颈识别所需工具：
  1. 结构化日志：每个阶段进入/退出带时间戳、请求 ID、检索文档数、上下文长度
  2. 阶段计时器 StageTimer：统计各阶段耗时
  3. cProfile 应用级性能分析
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
import uuid
from contextlib import contextmanager
from typing import Dict, List, Any


class StageTimer:
    """阶段计时器：记录每个 RAG 阶段的耗时。"""

    def __init__(self):
        self.stages: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = time.perf_counter() - t0

    def report(self) -> str:
        total = sum(self.stages.values())
        lines = [f"  总耗时 {total:.3f}s，各阶段："]
        for name, dt in self.stages.items():
            pct = dt / total * 100 if total else 0
            lines.append(f"    {name:<22} {dt:.3f}s ({pct:.1f}%)")
        return "\n".join(lines)


class StructuredLogger:
    """结构化日志：带时间戳、请求 ID、关键指标。"""

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def log(self, request_id: str, stage: str, **metrics):
        entry = {
            "ts": round(time.perf_counter(), 4),
            "request_id": request_id,
            "stage": stage,
            **metrics,
        }
        self.logs.append(entry)
        return entry

    def dump_json(self) -> str:
        return json.dumps(self.logs, ensure_ascii=False, indent=2)


def profile(fn, *args, **kwargs) -> str:
    """用 cProfile 分析函数调用，返回耗时统计文本。"""
    pr = cProfile.Profile()
    pr.enable()
    result = fn(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    return s.getvalue()


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]
