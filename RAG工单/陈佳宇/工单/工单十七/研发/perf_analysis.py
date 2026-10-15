# -*- coding: utf-8 -*-
# perf_analysis.py 工单17 内存对象分析辅助脚本
# 运行： python3 ‑m memory_profiler perf_analysis.py
import objgraph
import gc

@profile
def simulate_request_flow():
    """模拟一次完整请求生命周期，观测对象创建和残留"""
    big_data = [i for i in range(100000)]
    tmp_tensor = [0.1]*80000
    # 业务处理
    res = sum(big_data[:100])

    # =========优化前：不手动释放对象========
    # =========优化后：请求结束主动释放大对象========
    del big_data
    del tmp_tensor
    gc.collect()
    return res


def show_leak_object():
    print("\n====objgraph 打印增长最多对象类型====")
    objgraph.show_growth(limit=15)


if __name__ == "__main__":
    for _ in range(10):
        simulate_request_flow()
    show_leak_object()
