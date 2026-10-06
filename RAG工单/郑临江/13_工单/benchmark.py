# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
基准测试：单独对各组件（检索、LLM 推理、嵌入）做原始性能测量。
"""
import time

import config
from pipeline import RAGPipeline


def benchmark_retrieve(pipeline, queries, n=100):
    """检索组件基准：多次查询取平均耗时。"""
    q_tokens = [pipeline._build_index if False else None]  # noqa
    start = time.perf_counter()
    for _ in range(n):
        for q in queries:
            pipeline.retrieve(__import__("pipeline").tokenize(q))
    elapsed = time.perf_counter() - start
    return elapsed / (n * len(queries))


def benchmark_llm(n=20):
    """LLM 推理组件基准（离线模拟）。"""
    start = time.perf_counter()
    for _ in range(n):
        time.sleep(config.SIMULATED_LLM_LATENCY)
    return (time.perf_counter() - start) / n


def run_benchmarks(pipeline, queries):
    print("===== 组件基准测试 =====")
    t_ret = benchmark_retrieve(pipeline, queries)
    print(f"检索组件平均耗时：{t_ret * 1000:.1f} ms/次")
    t_llm = benchmark_llm()
    print(f"LLM 推理平均耗时：{t_llm * 1000:.1f} ms/次")
    print("提示：可根据各组件原始性能特征，选择更快的嵌入模型 / 向量库 / 推理配置。")
