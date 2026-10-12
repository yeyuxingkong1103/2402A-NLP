# -*- coding: utf-8 -*-
"""
app.py — 金融问答系统（Flask HTTP 服务）
工单编号：人工智能NLP-RAG-金融问答系统部署

以容器化部署为目标的最小 HTTP 服务：
  - GET  /health       健康检查
  - POST /ask          问答：{"query": "..."} → {"answer": "...", "sources": [...]}
  - POST /index/build  重建索引（数据更新后调用）
"""
from __future__ import annotations

import glob
import os
import sys

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (Config, VectorStore, BM25Index, split_text, make_embedder,
                      make_llm, answer, HybridRetriever)  # noqa: E402

# 数据目录（Docker 卷挂载点，默认 ./data，可被 -v 覆盖）
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

app = Flask(__name__)
_config = Config()
_llm = make_llm(_config)
_retriever = None
_chunks = []


def build_index():
    global _retriever, _chunks
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)
    if not files:
        # 兼容直接指向 ccf txt 目录
        files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    chunks = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            text = fp.read()
        name = os.path.basename(f)
        chunks.extend([f"[{name}] {p}" for p in split_text(text, 400, 60)])
    _chunks = chunks
    embedder = make_embedder(_config.embedding_model, chunks)
    _retriever = HybridRetriever(VectorStore(embedder, chunks), BM25Index(chunks))
    return len(chunks)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "chunks": len(_chunks)})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query 不能为空"}), 400
    if _retriever is None:
        return jsonify({"error": "索引未构建，请先 POST /index/build"}), 503
    hits = _retriever.retrieve(query, top_k=5)
    ans = answer(query, hits, _llm)
    return jsonify({
        "answer": ans,
        "sources": [{"score": h["score"], "text": h["text"][:200]} for h in hits[:3]],
    })


@app.route("/index/build", methods=["POST"])
def index_build():
    n = build_index()
    return jsonify({"status": "ok", "chunks": n})


if __name__ == "__main__":
    n = build_index()
    print(f"索引构建完成：{n} 块")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
