# -*- coding: utf-8 -*-
"""
Web 应用入口 V5 - 多轮对话聊天界面
工单编号: 人工智能 NLP-RAG-Query 理解优化任务

启动: python app_v5.py
访问: http://127.0.0.1:5004
"""
import os, sys, logging, uuid, json
from flask import Flask, request, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v5 as config
import qa_engine_v5

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v5", static_folder="static")


@app.route("/")
def index():
    return render_template("index_v5.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    q = (data.get("question") or "").strip()
    sid = data.get("session_id") or request.cookies.get("rag_session") or str(uuid.uuid4())
    if not q:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        result = qa_engine_v5.answer_question(q, sid)
        resp = jsonify(result)
        resp.set_cookie("rag_session", sid, max_age=86400)
        return resp
    except Exception as e:
        logging.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/new_session", methods=["POST"])
def api_new_session():
    sid = request.get_json().get("session_id") or str(uuid.uuid4())
    result = qa_engine_v5.new_session(sid)
    resp = jsonify(result)
    resp.set_cookie("rag_session", sid, max_age=86400)
    return resp


@app.route("/api/context", methods=["GET"])
def api_context():
    sid = request.args.get("session_id") or request.cookies.get("rag_session", "default")
    return jsonify(qa_engine_v5.get_session_context(sid))


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "version": "v5", "port": config.PORT})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=False)
