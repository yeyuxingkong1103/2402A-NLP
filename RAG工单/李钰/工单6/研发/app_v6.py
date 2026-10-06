# -*- coding: utf-8 -*-
"""
Web 应用入口 V6 - 混合检索配置化
工单编号: 人工智能 NLP-RAG-混合检索任务

启动: python app_v6.py
访问: http://127.0.0.1:5005
"""
import os, sys, logging
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v6 as config
import qa_engine_v6

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v6")


@app.route("/")
def index():
    return render_template("index_v6.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    q = (data.get("question") or "").strip()
    strategy = data.get("strategy")  # vector / fulltext / hybrid
    if not q:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        result = qa_engine_v6.answer_question(q, strategy=strategy)
        return jsonify(result)
    except Exception as e:
        logging.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """用户反馈: 👍/👎 对某个 chunk 的"""
    data = request.get_json() or {}
    chunk_id = str(data.get("chunk_id", ""))
    is_positive = bool(data.get("positive", True))
    if not chunk_id:
        return jsonify({"error": "chunk_id 不能为空"}), 400
    result = qa_engine_v6.record_feedback(chunk_id, is_positive)
    return jsonify(result)


@app.route("/api/config")
def api_config():
    return jsonify(qa_engine_v6.get_config())


@app.route("/api/config", methods=["POST"])
def api_update_config():
    """动态更新配置 (不重启)"""
    data = request.get_json() or {}
    global config
    if "strategy" in data and data["strategy"] in ("vector", "fulltext", "hybrid"):
        config.RETRIEVAL_STRATEGY = data["strategy"]
    if "vector_weight" in data:
        config.VECTOR_WEIGHT = float(data["vector_weight"])
    if "fulltext_weight" in data:
        config.FULLTEXT_WEIGHT = float(data["fulltext_weight"])
    if "fusion_method" in data and data["fusion_method"] in ("weighted", "rrf", "vote"):
        config.FUSION_METHOD = data["fusion_method"]
    return jsonify(qa_engine_v6.get_config())


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "version": "v6", "port": config.PORT,
                    "config": qa_engine_v6.get_config()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=False)
