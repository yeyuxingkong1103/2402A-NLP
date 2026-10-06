# -*- coding: utf-8 -*-
"""
Web 应用入口 V9 - Graph RAG 优化 + RAGAS 评估 + V8 vs V9 对比
工单编号: 人工智能 NLP-RAG-Graph RAG 优化任务

启动: python app_v9.py
访问: http://127.0.0.1:5008
"""
import os, sys, logging
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_engine_v9

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v9")


@app.route("/")
def index():
    return render_template("index_v9.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        result = qa_engine_v9.answer_question(q)
        return jsonify(result)
    except Exception as e:
        logging.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """V8 vs V9 对比"""
    data = request.get_json() or {}
    suite = data.get("test_suite") or [
        {"question": "武汉力源信息技术股份有限公司的控股股东是谁?",
         "ref_keywords": ["武汉力源科技", "35", "控股股东"]},
        {"question": "销售部有几个下属部门,大客户销售部有几个销售处?",
         "ref_keywords": ["销售部", "大客户销售部", "销售处"]},
        {"question": "武汉兴图新科参与制定了什么标准?",
         "ref_keywords": ["AVS", "标准"]},
        {"question": "武汉兴图新科的注册资本是多少?",
         "ref_keywords": ["7360", "注册资本"]},
        {"question": "军用领域收入分别是多少?",
         "ref_keywords": ["军用", "收入"]},
    ]
    try:
        result = qa_engine_v9.run_v8_v9_comparison(suite)
        return jsonify(result)
    except Exception as e:
        logging.exception("对比失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kg")
def api_kg():
    _ensure = qa_engine_v9._ensure_ready()
    return jsonify(qa_engine_v9._KG.to_visualization())


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "version": "v9", "port": 5008,
                    "targets": {"context_precision": 0.80, "context_recall": 0.90}})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, debug=False)
