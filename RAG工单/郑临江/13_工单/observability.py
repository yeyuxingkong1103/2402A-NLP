# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
可观测性：结构化日志（含请求ID、时间戳、阶段指标）+ Prometheus 监控指标。
"""
import logging
import time
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag")


def log_stage(request_id, query, stages, total):
    """结构化日志：每个阶段进入/退出带详细时间戳。"""
    logger.info(
        "[request_id=%s] query=%s | 查询处理=%.3fs 检索=%.3fs 上下文=%.3fs "
        "LLM=%.3fs 后处理=%.3fs | 总耗时=%.3fs",
        request_id, query[:20],
        stages.get("1_query_process", 0),
        stages.get("2_retrieve", 0),
        stages.get("3_context", 0),
        stages.get("4_llm", 0),
        stages.get("5_postprocess", 0),
        total,
    )


class Metrics:
    """Prometheus 风格指标：延迟、吞吐量、错误率、队列长度。"""

    def __init__(self):
        self.latencies = deque(maxlen=1000)
        self.errors = 0
        self.requests = 0
        self.queue_len = 0
        self.start = time.time()

    def record(self, total, error=False):
        self.latencies.append(total)
        self.requests += 1
        if error:
            self.errors += 1

    def _pct(self, p):
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = min(int(len(s) * p), len(s) - 1)
        return s[idx]

    def report(self):
        elapsed = max(time.time() - self.start, 1)
        n = max(len(self.latencies), 1)
        return {
            "latency_avg": sum(self.latencies) / n,
            "latency_p95": self._pct(0.95),
            "latency_p99": self._pct(0.99),
            "throughput_rps": self.requests / elapsed,
            "error_rate": self.errors / max(self.requests, 1),
            "queue_len": self.queue_len,
        }

    def prometheus_text(self):
        r = self.report()
        return "\n".join([
            "# TYPE rag_latency_avg gauge",
            f"rag_latency_avg {r['latency_avg']:.3f}",
            "# TYPE rag_latency_p95 gauge",
            f"rag_latency_p95 {r['latency_p95']:.3f}",
            "# TYPE rag_throughput_rps gauge",
            f"rag_throughput_rps {r['throughput_rps']:.3f}",
            "# TYPE rag_error_rate gauge",
            f"rag_error_rate {r['error_rate']:.3f}",
            "# TYPE rag_queue_len gauge",
            f"rag_queue_len {r['queue_len']}",
        ])


METRICS = Metrics()
