# -*- coding: utf-8 -*-
"""
Web 应用入口 V8 - Graph RAG + 图谱可视化
工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答

启动: python app_v8.py
访问: http://127.0.0.1:5007
"""
import os, sys, logging
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_engine_v8

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v8")


@app.route("/")
def index():
    return render_template("index_v8.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        result = qa_engine_v8.answer_question(q)
        return jsonify(result)
    except Exception as e:
        logging.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kg")
def api_kg():
    """获取完整知识图谱"""
    try:
        return jsonify(qa_engine_v8.get_kg_visualization())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/kg_query", methods=["POST"])
def api_kg_query():
    """直接查询知识图谱"""
    data = request.get_json() or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        return jsonify(qa_engine_v8.query_graph_directly(q))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "version": "v8", "port": 5007})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, debug=False)
