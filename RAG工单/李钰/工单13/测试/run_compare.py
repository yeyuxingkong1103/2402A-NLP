# -*- coding: utf-8 -*-
"""
RAG 性能优化: 优化前后对比脚本 (验收用)
工单编号: 人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化

用法: python run_compare.py
"""
import os, sys, json, time, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 确保研发目录可 import
_dev = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "研发")
sys.path.insert(0, _dev)

logging.basicConfig(level=logging.WARNING)

from performance_monitor import (
    TraceContext, get_metrics_summary, generate_report, reset_metrics
)
from baseline_rag import BaselineRAG
from optimized_rag import OptimizedRAG
from bottleneck_identifier import diagnose_bottlenecks
import config_v13 as config


# 测试问题集 (覆盖各类型)
TEST_QUERIES = [
    "武汉力源的控股股东是谁?",
    "武汉力源的销售部有几个部门?",
    "武汉力源的募集资金投资哪些项目?",
    "武汉兴图新科的注册资本是多少?",
    "武汉兴图新科参与制定了什么技术标准?",
    "军用领域收入分别是多少?",
    "IC 市场增长最快的行业?",
    "武汉力源发行股数是多少?",
]


def run_single(engine_cls, queries: list, label: str) -> dict:
    """运行单个引擎"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # 预热
    warmup = TraceContext("warmup")
    engine = engine_cls(warmup)
    engine.query("warmup")

    results = []
    for i, q in enumerate(queries):
        trace = TraceContext(f"{label}_{i}")
        engine = engine_cls(trace)
        r = engine.query(q)
        trace.save_trace()
        results.append({
            "query": q,
            "total_ms": r["total_ms"],
            "trace": trace.to_dict(),
        })
        print(f"  [{i+1}] {q[:25]:25s} → {r['total_ms']:>7.0f}ms  "
              f"({r['total_ms']/1000:.2f}s)")

    avg = sum(r["total_ms"] for r in results) / len(results)
    max_ms = max(r["total_ms"] for r in results)
    print(f"\n  平均: {avg:.0f}ms  ({avg/1000:.2f}s)")
    print(f"  最大: {max_ms:.0f}ms  ({max_ms/1000:.2f}s)")
    print(f"  验收: {'✅ 通过' if avg < config.MAX_LATENCY_MS else '❌ 未通过'} "
          f"(< {config.MAX_LATENCY_MS}ms)")

    return {"label": label, "results": results, "avg": avg, "max": max_ms}


def main():
    reset_metrics()

    print("=" * 60)
    print("  RAG V13 性能优化 - 优化前后对比")
    print("  工单编号: 人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化")
    print(f"  验收目标: < {config.MAX_LATENCY_MS}ms")
    print("=" * 60)

    # 1. 基线
    baseline = run_single(BaselineRAG, TEST_QUERIES, "Baseline RAG (未优化)")

    # 2. 优化后
    optimized = run_single(OptimizedRAG, TEST_QUERIES, "Optimized FastRAG (6 大优化)")

    # 3. 汇总对比
    print(f"\n{'='*60}")
    print("  📊 优化前后对比汇总")
    print(f"{'='*60}")

    improvement_pct = (baseline["avg"] - optimized["avg"]) / baseline["avg"] * 100
    print(f"  基线平均:    {baseline['avg']:>8.0f}ms  ({baseline['avg']/1000:.2f}s)")
    print(f"  优化后平均:  {optimized['avg']:>8.0f}ms  ({optimized['avg']/1000:.2f}s)")
    print(f"  延迟降低:    ↓ {improvement_pct:.1f}%")
    print(f"  验收:        {'✅ 通过' if optimized['avg'] < config.MAX_LATENCY_MS else '❌ 未通过'} "
          f"(< {config.MAX_LATENCY_MS}ms)")

    # 4. 瓶颈诊断
    summary = get_metrics_summary()
    bottlenecks = diagnose_bottlenecks(summary)

    print(f"\n{'='*60}")
    print("  🔍 瓶颈诊断")
    print(f"{'='*60}")

    if bottlenecks:
        for b in bottlenecks:
            print(f"  {b['severity']} [{b['type']}] {b['stage']:20s} mean={b['mean_ms']:>7.0f}ms "
                  f"(阈值 {b['threshold_ms']}ms)")
            print(f"      优化建议: {', '.join(b['recommendation'])}")
    else:
        print("  ✅ 无明显瓶颈")

    # 5. 详细报告
    print(f"\n{generate_report()}")

    # 6. 保存 JSON
    output = {
        "acceptance_target_ms": config.MAX_LATENCY_MS,
        "baseline": {
            "avg_ms": round(baseline["avg"], 2),
            "max_ms": round(baseline["max"], 2),
            "pass": baseline["avg"] < config.MAX_LATENCY_MS,
        },
        "optimized": {
            "avg_ms": round(optimized["avg"], 2),
            "max_ms": round(optimized["max"], 2),
            "pass": optimized["avg"] < config.MAX_LATENCY_MS,
        },
        "improvement_pct": round(improvement_pct, 2),
        "bottlenecks": bottlenecks,
        "per_query": [
            {
                "query": TEST_QUERIES[i],
                "baseline_ms": baseline["results"][i]["total_ms"],
                "optimized_ms": optimized["results"][i]["total_ms"],
            }
            for i in range(len(TEST_QUERIES))
        ],
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "optimization_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  详细数据 → {out_path}")

    return output


if __name__ == "__main__":
    main()
