# -*- coding: utf-8 -*-
"""
性能监控 + TraceContext (分布式追踪简化版)
工单编号: 人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化

特性:
  1. 每个阶段 enter/exit 自动计时
  2. request_id 贯穿全流程
  3. 结构化日志 + 指标采集
  4. cProfile 集成
"""
import time, uuid, threading, logging, json
from contextlib import contextmanager
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============ 全局指标存储 ============
_metrics_lock = threading.Lock()
_metrics: Dict[str, List[float]] = defaultdict(list)
_request_traces: Dict[str, Dict] = {}


def record_metric(name: str, value: float):
    """记录指标"""
    with _metrics_lock:
        _metrics[name].append(value)


def get_metrics_summary() -> Dict:
    """获取汇总"""
    result = {}
    with _metrics_lock:
        for name, values in _metrics.items():
            if not values:
                continue
            values_sorted = sorted(values)
            n = len(values)
            result[name] = {
                "count": n,
                "mean": round(sum(values) / n, 4),
                "median": round(values_sorted[n // 2], 4),
                "p95": round(values_sorted[int(n * 0.95)], 4) if n >= 20 else None,
                "max": round(max(values), 4),
                "min": round(min(values), 4),
            }
    return result


def reset_metrics():
    with _metrics_lock:
        _metrics.clear()
        _request_traces.clear()


# ============ TraceContext ============

class TraceContext:
    """请求追踪上下文"""

    def __init__(self, request_id: str = None):
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self.start_time = time.time()
        self.stages: List[Dict] = []
        self.current_stage: Optional[Dict] = None

    def enter_stage(self, name: str, **kwargs):
        """进入阶段"""
        self.current_stage = {
            "name": name,
            "start": time.time(),
            "extra": kwargs,
        }
        logger.info(f"[{self.request_id}] → {name}")

    def exit_stage(self, result_info: str = ""):
        """退出阶段"""
        if self.current_stage:
            elapsed = time.time() - self.current_stage["start"]
            self.current_stage["end"] = time.time()
            self.current_stage["elapsed_ms"] = round(elapsed * 1000, 2)
            self.current_stage["result"] = result_info
            self.stages.append(self.current_stage)

            # 记录到全局指标
            record_metric(f"stage.{self.current_stage['name']}", elapsed * 1000)

            logger.info(
                f"[{self.request_id}] ← {self.current_stage['name']} "
                f"({elapsed*1000:.1f}ms) {result_info}"
            )
            self.current_stage = None

    def total_time_ms(self) -> float:
        return round((time.time() - self.start_time) * 1000, 2)

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "total_ms": self.total_time_ms(),
            "stages": self.stages,
            "bottleneck": self._find_bottleneck(),
        }

    def _find_bottleneck(self) -> Optional[Dict]:
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: s["elapsed_ms"])

    def save_trace(self):
        _request_traces[self.request_id] = self.to_dict()


# ============ 装饰器 ============

def timed_stage(name: str):
    """阶段计时装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 尝试从第一个参数找 trace
            trace = None
            if args and hasattr(args[0], "trace"):
                trace = args[0].trace
            if trace:
                trace.enter_stage(name)
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                if trace:
                    trace.exit_stage()
                record_metric(f"stage.{name}", elapsed)
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                if trace:
                    trace.exit_stage(f"ERROR: {e}")
                raise
        return wrapper
    return decorator


# ============ cProfile 工具 ============

def profile_function(func, *args, top_n: int = 15, **kwargs):
    """对函数做 cProfile 分析"""
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(top_n)

    print(f"\n=== cProfile: {func.__name__} (Top {top_n}) ===")
    print(s.getvalue())
    return result


# ============ 报告生成 ============

def generate_report() -> str:
    """生成性能分析报告"""
    summary = get_metrics_summary()
    lines = ["=" * 60, "  RAG 性能分析报告", "=" * 60]

    for name, stats in sorted(summary.items()):
        lines.append(
            f"  {name:<30} count={stats['count']:>4}  "
            f"mean={stats['mean']:>7.1f}ms  "
            f"median={stats['median']:>7.1f}ms  "
            f"max={stats['max']:>7.1f}ms"
        )

    lines.append("\n=== 最慢阶段 TOP 5 ===")
    stage_stats = [(name, s) for name, s in summary.items() if name.startswith("stage.")]
    stage_stats.sort(key=lambda x: x[1]["mean"], reverse=True)
    for i, (name, s) in enumerate(stage_stats[:5], 1):
        lines.append(f"  {i}. {name:<28} mean={s['mean']:.1f}ms")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 模拟 10 个请求
    for i in range(10):
        trace = TraceContext(f"req_{i}")
        trace.enter_stage("query_process")
        time.sleep(0.05)
        trace.exit_stage()

        trace.enter_stage("retrieve")
        time.sleep(0.3)
        trace.exit_stage("found 5 chunks")

        trace.enter_stage("llm_generate")
        time.sleep(1.5)
        trace.exit_stage("tokens=200")

        trace.save_trace()

    print(generate_report())
