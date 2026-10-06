# -*- coding: utf-8 -*-
"""
Web 应用入口 (Flask)
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

启动方式:
    python app.py
然后浏览器访问: http://127.0.0.1:5000
"""
import os
import logging
from flask import Flask, request, jsonify, render_template

# 使本目录下的模块可被导入
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import qa_engine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question 不能为空"}), 400
    try:
        result = qa_engine.answer_question(question)
        # 过滤 retrieval 中的长文本避免响应过大
        for r in result.get("retrieval", []):
            r["text"] = r["text"][:300]
        return jsonify(result)
    except Exception as e:
        logger.exception("问答失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "pdf_path": config.PDF_PATH,
        "pdf_exists": os.path.exists(config.PDF_PATH),
        "llm_configured": bool(config.LLM_API_KEY),
    })


@app.route("/api/rebuild", methods=["POST"])
def api_rebuild():
    """重建索引"""
    import vector_retrieval
    import pdf_parser
    chunks = pdf_parser.parse_and_save()
    retriever = vector_retrieval.VectorRetriever()
    retriever.build_index(chunks)
    retriever.save_index()
    return jsonify({"rebuilt": True, "chunks": len(chunks)})


if __name__ == "__main__":
    # 启动前预热 (加载索引)
    try:
        qa_engine._get_retriever()
    except Exception as e:
        logger.warning(f"启动时索引未就绪: {e}")
    app.run(host="0.0.0.0", port=5000, debug=False)
