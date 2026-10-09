# -*- coding: utf-8 -*-
"""
V13 Flask 入口 + Prometheus 风格指标
工单编号: 人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化
端口: 5010
"""
import os, sys, json, time, logging
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v13 as config
from performance_monitor import (
    TraceContext, get_metrics_summary, generate_report, reset_metrics
)
from baseline_rag import BaselineRAG
from optimized_rag import OptimizedRAG
from bottleneck_identifier import diagnose_bottlenecks

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v13")

# 预热 + 全局实例
optimized_engine = OptimizedRAG()


@app.route("/")
def index():
    return render_template("index_v13.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "optimized")  # baseline / optimized / both

    if not question:
        return jsonify({"error": "question required"}), 400

    result = {"question": question}

    if mode in ("baseline", "both"):
        trace_b = TraceContext()
        rag_b = BaselineRAG(trace_b)
        t0 = time.time()
        r = rag_b.query(question)
        trace_b.save_trace()
        result["baseline"] = {
            "answer": r["answer"],
            "total_ms": r["total_ms"],
            "stages": trace_b.to_dict(),
        }

    if mode in ("optimized", "both"):
        trace_o = TraceContext()
        rag_o = OptimizedRAG(trace_o)
        r = rag_o.query(question)
        trace_o.save_trace()
        result["optimized"] = {
            "answer": r["answer"],
            "total_ms": r["total_ms"],
            "cache_hit": r.get("cache_hit", False),
            "stages": trace_o.to_dict(),
        }

    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    """Prometheus 风格指标"""
    summary = get_metrics_summary()
    bottlenecks = diagnose_bottlenecks(summary)

    # 计算验收通过率
    baseline_etoe = summary.get("end_to_end.baseline", {}).get("mean", 0)
    opt_etoe = summary.get("end_to_end.optimized", {}).get("mean", 0)
    pass_rate = opt_etoe > 0 and opt_etoe < config.MAX_LATENCY_MS

    return jsonify({
        "summary": summary,
        "bottlenecks": bottlenecks,
        "acceptance": {
            "baseline_avg_ms": baseline_etoe,
            "optimized_avg_ms": opt_etoe,
            "target_ms": config.MAX_LATENCY_MS,
            "pass": pass_rate,
            "improvement_pct": round(
                (baseline_etoe - opt_etoe) / baseline_etoe * 100, 1
            ) if baseline_etoe > 0 else 0,
        },
    })


@app.route("/api/report")
def api_report():
    text = generate_report()
    return text, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_metrics()
    return jsonify({"ok": True})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5010)
    args = parser.parse_args()
    print(f"\nV13 RAG 性能优化: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
