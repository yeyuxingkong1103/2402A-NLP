# -*- coding: utf-8 -*-
"""
Web 应用入口 V7 - 评估可视化
工单编号: 人工智能 NLP-RAG-功能测试及评估

启动: python app_v7.py
访问: http://127.0.0.1:5006
"""
import os, sys, logging
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_engine_v7

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v7")


@app.route("/")
def index():
    return render_template("index_v7.html")


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    try:
        result = qa_engine_v7.run_full_evaluation()
        # 序列化 (移除不可 JSON 化的内容)
        summary = {
            "retrieval_metrics": result["retrieval_metrics"],
            "qa_metrics": result["qa_metrics"],
            "perf_metrics": result["perf_metrics"],
            "problem_count": len(result["problem_analysis"]),
            "problem_analysis": result["problem_analysis"],
            "results": [{
                "id": r["id"], "question": r["question"],
                "type": r["type"], "difficulty": r["difficulty"],
                "retrieval": r["retrieval"], "qa": r["qa"],
                "response_time": r["response_time"],
                "rag_answer": r["rag_answer"][:200],
            } for r in result["results"]],
        }
        return jsonify(summary)
    except Exception as e:
        logging.exception("评估失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate_single", methods=["POST"])
def api_evaluate_single():
    data = request.get_json() or {}
    qid = data.get("id", 1)
    suite = qa_engine_v7.test_suite.get_test_suite()
    item = next((q for q in suite if q["id"] == qid), None)
    if not item:
        return jsonify({"error": "问题不存在"}), 404
    try:
        result = qa_engine_v7.run_single_evaluation(item)
        return jsonify(result)
    except Exception as e:
        logging.exception("单题评估失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "version": "v7", "port": 5006})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=False)
