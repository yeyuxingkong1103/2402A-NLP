# -*- coding: utf-8 -*-
"""
Web 应用入口 V3 (Flask)
工单编号: 人工智能 NLP-RAG-PDF 文档的表格解析及检索优化

启动:
    python app_v3.py
访问:
    http://127.0.0.1:5002 (V3 独立端口)
"""
import os
import sys
import logging
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_v3 as config
import qa_engine_v3

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates_v3")


@app.route("/")
def index():
    return render_template("index_v3.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        result = qa_engine_v3.answer_question(question)
        # 截断表格行文本避免过大
        for tr in result.get("table_results", []):
            for hr in tr.get("hit_rows", []):
                hr["row"] = hr["row"][:50]
        for tr in result.get("text_results", []):
            if isinstance(tr, dict) and "text" in tr:
                tr["text"] = tr["text"][:300]
        return jsonify(result)
    except Exception as e:
        logging.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    docs_status = []
    for spec in config.PDF_DOCS:
        path = spec["path"]
        docs_status.append({
            "company": spec["company"],
            "path": path,
            "exists": os.path.exists(path),
            "uses_preset": spec["company"] == "武汉力源信息技术股份有限公司"
                           and not os.path.exists(path),
        })
    return jsonify({
        "status": "ok",
        "version": "v3",
        "llm_configured": bool(config.LLM_API_KEY),
        "documents": docs_status,
    })


if __name__ == "__main__":
    try:
        qa_engine_v3._ensure_ready()
    except Exception as e:
        logging.warning(f"启动时索引未就绪: {e}")
    app.run(host="0.0.0.0", port=5002, debug=False)
