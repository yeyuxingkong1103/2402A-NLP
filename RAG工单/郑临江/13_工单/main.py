# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
主程序：分析检索性能慢的原因并优化，输出优化前后检索时间对比。
"""
import statistics
import time

import config
from pipeline import RAGPipeline
from observability import METRICS
from profiling import profile_func, snakeviz_hint
from benchmark import run_benchmarks


def load_docs():
    import pymupdf
    docs = []
    for p in [config.PDF1, config.PDF2]:
        doc = pymupdf.open(p)
        text = "\n".join(pg.get_text() for pg in doc)
        doc.close()
        # 简单切块
        size, overlap = 500, 80
        start = 0
        while start < len(text):
            docs.append({"id": f"{p}-{start}", "text": text[start:start + size]})
            start += size - overlap
    return docs


def measure(pipeline, queries):
    totals = []
    for q in queries:
        r = pipeline.run(q)
        totals.append(r["total"])
    return statistics.mean(totals), totals


def main():
    docs = load_docs()
    print(f"[main] 加载文档块 {len(docs)} 个")

    # ===== 优化前 =====
    print("\n===== 优化前（未建立索引、无缓存）=====")
    before_pipe = RAGPipeline(docs, pre_index=False, cache=False)
    before_avg, before_times = measure(before_pipe, config.TEST_QUERIES)
    print(f"优化前平均检索耗时：{before_avg * 1000:.1f} ms")
    for q, t in zip(config.TEST_QUERIES, before_times):
        print(f"  [{t * 1000:.1f}ms] {q[:24]}")

    # ===== 优化后 =====
    print("\n===== 优化后（预建索引 + 结果缓存）=====")
    after_pipe = RAGPipeline(docs, pre_index=True, cache=True)
    after_avg, after_times = measure(after_pipe, config.TEST_QUERIES)
    # 第二轮重复查询命中缓存，进一步降低延迟
    after_avg2, _ = measure(after_pipe, config.TEST_QUERIES)
    print(f"优化后平均检索耗时（首轮）：{after_avg * 1000:.1f} ms")
    print(f"优化后平均检索耗时（缓存命中）：{after_avg2 * 1000:.1f} ms")

    # ===== 对比 =====
    print("\n===== 优化前后对比 =====")
    speedup = before_avg / max(after_avg, 1e-6)
    print(f"平均耗时：{before_avg * 1000:.1f} ms -> {after_avg * 1000:.1f} ms")
    print(f"加速比：{speedup:.1f}x")
    print(f"缓存命中后：{after_avg2 * 1000:.1f} ms")
    print(f"是否满足 <= {config.TARGET_LATENCY}s 验收标准："
          f"{'是' if after_avg <= config.TARGET_LATENCY else '否'}")

    # ===== 阶段耗时定位瓶颈 =====
    print("\n===== 瓶颈分析（优化后各阶段耗时）=====")
    sample = after_pipe.run(config.TEST_QUERIES[0])
    total = sample["total"]
    for k, v in sample["stages"].items():
        bar = "#" * int(v / max(total, 1e-6) * 40)
        print(f"{k}: {v * 1000:7.1f} ms ({v / total * 100:5.1f}%) {bar}")

    # ===== 监控指标 =====
    print("\n===== 监控指标（Prometheus 文本格式）=====")
    print(METRICS.prometheus_text())

    # ===== 性能分析（cProfile + snakeviz）=====
    print("\n===== 应用级性能分析 =====")
    profile_func(after_pipe.run, config.TEST_QUERIES[0])
    snakeviz_hint()

    # ===== 基准测试 =====
    run_benchmarks(after_pipe, config.TEST_QUERIES)


if __name__ == "__main__":
    main()
