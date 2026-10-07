# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
模块：RAG 性能分析器
功能：为每个阶段加时间戳日志，输出耗时明细
"""

import time
import json
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List


class RAGProfiler:
    """RAG 性能分析器"""

    def __init__(self, name: str = "rag_profiler"):
        self.name = name
        self.stages: Dict[str, List[float]] = defaultdict(list)
        self.current: Dict[str, float] = {}

    @contextmanager
    def stage(self, stage_name: str):
        """上下文管理器：自动记录阶段耗时"""
        t0 = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - t0
            self.stages[stage_name].append(elapsed)

    def record(self, stage_name: str, elapsed: float):
        """手动记录"""
        self.stages[stage_name].append(elapsed)

    def start(self, stage_name: str):
        self.current[stage_name] = time.time()

    def end(self, stage_name: str) -> float:
        if stage_name in self.current:
            elapsed = time.time() - self.current[stage_name]
            self.stages[stage_name].append(elapsed)
            del self.current[stage_name]
            return elapsed
        return 0.0

    def summary(self) -> Dict:
        """汇总统计"""
        result = {}
        for stage, times in self.stages.items():
            result[stage] = {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }
        return result

    def print_summary(self):
        print("=" * 80)
        print(f"【{self.name}】性能汇总")
        print("=" * 80)
        print(f"{'阶段':<30} {'次数':<6} {'平均':<10} {'最小':<10} {'最大':<10}")
        print("-" * 80)
        for stage, stats in self.summary().items():
            print(f"{stage:<30} {stats['count']:<6} {stats['avg']:<10.4f} "
                  f"{stats['min']:<10.4f} {stats['max']:<10.4f}")

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存：{path}")


# 全局单例
_profiler = RAGProfiler()


def get_profiler() -> RAGProfiler:
    return _profiler
