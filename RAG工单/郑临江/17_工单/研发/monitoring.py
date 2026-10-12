# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏工单
Prometheus 监控：请求量、延迟直方图、队列长度、内存/GPU 占用。
"""
import os
import time
from collections import deque


class PrometheusMetrics:
    def __init__(self):
        self.latencies = deque(maxlen=10000)
        self.req_total = 0
        self.err_total = 0
        self.start = time.time()

    def observe(self, latency, error=False):
        self.latencies.append(latency)
        self.req_total += 1
        if error:
            self.err_total += 1

    def queue_len(self):
        import queue as q
        return getattr(q, "_queue_len", 0)

    def _pct(self, p):
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[min(int(len(s) * p), len(s) - 1)]

    def mem_usage(self):
        """当前进程 RSS（MB），用于泄漏检测。"""
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def gpu_usage(self):
        """GPU 显存占用（MB），通过 nvidia-smi 采集。"""
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                timeout=5).decode()
            return int(out.strip().split()[0])
        except Exception:
            return 0.0

    def render(self):
        elapsed = max(time.time() - self.start, 1)
        n = max(len(self.latencies), 1)
        lines = [
            "# HELP rag_request_total 请求总数",
            "# TYPE rag_request_total counter",
            f"rag_request_total {self.req_total}",
            "# TYPE rag_latency_p95 gauge",
            f"rag_latency_p95 {self._pct(0.95):.3f}",
            "# TYPE rag_latency_p99 gauge",
            f"rag_latency_p99 {self._pct(0.99):.3f}",
            "# TYPE rag_throughput_rps gauge",
            f"rag_throughput_rps {self.req_total / elapsed:.3f}",
            "# TYPE rag_error_rate gauge",
            f"rag_error_rate {self.err_total / max(self.req_total, 1):.3f}",
            "# TYPE rag_process_rss_mb gauge",
            f"rag_process_rss_mb {self.mem_usage():.1f}",
            "# TYPE rag_gpu_mem_mb gauge",
            f"rag_gpu_mem_mb {self.gpu_usage()}",
        ]
        return "\n".join(lines)


METRICS = PrometheusMetrics()
