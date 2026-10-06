# -*- coding: utf-8 -*-
"""
Web 应用入口 V2 (Flask)
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

启动:
    python app_v2.py
访问:
    http://127.0.0.1:5001 (V2 使用不同端口避免与 V1 冲突)
"""
import os
import sys
import logging
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_v2 as config
import qa_engine_v2

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v2", static_folder="static")


@app.route("/")
def index():
    return render_template("index_v2.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        result = qa_engine_v2.answer_question(question)
        for r in result.get("retrieval", []):
            r["text"] = r["text"][:300]
        return jsonify(result)
    except Exception as e:
        logging.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "version": "v2",
        "pdf_path": config.PDF_PATH,
        "pdf_exists": os.path.exists(config.PDF_PATH),
        "llm_configured": bool(config.LLM_API_KEY),
    })


@app.route("/api/rebuild", methods=["POST"])
def api_rebuild():
    import pdf_parser_v2
    import vector_retrieval_v2
    chunks = pdf_parser_v2.parse_and_save_v2()
    retriever = vector_retrieval_v2.HybridRetriever()
    retriever.build_index(chunks)
    retriever.save_index()
    return jsonify({"rebuilt": True, "chunks": len(chunks)})


if __name__ == "__main__":
    try:
        qa_engine_v2._get_retriever()
    except Exception as e:
        logging.warning(f"启动时索引未就绪: {e}")
    app.run(host="0.0.0.0", port=5001, debug=False)
