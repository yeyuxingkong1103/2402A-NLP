# -*- coding: utf-8 -*-
"""
Web 应用入口 V4 (Flask)
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

启动: python app_v4.py
访问: http://127.0.0.1:5003
"""
import os, sys, logging
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v4 as config
import qa_engine_v4

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v4")


@app.route("/")
def index():
    return render_template("index_v4.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        result = qa_engine_v4.answer_question(q)
        return jsonify(result)
    except Exception as e:
        logging.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "version": "v4",
        "clip_available": config.CLIP_AVAILABLE,
        "llm_configured": bool(config.LLM_API_KEY),
        "vision_configured": bool(config.VISION_API_KEY),
    })


if __name__ == "__main__":
    try:
        qa_engine_v4._ensure_ready()
    except Exception as e:
        logging.warning(f"启动时索引未就绪: {e}")
    app.run(host="0.0.0.0", port=5003, debug=False)
