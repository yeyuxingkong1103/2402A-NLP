# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-金融问答系统部署
Flask 服务：提供金融问答 HTTP 接口与简单界面。
接口：
  GET  /          -> 问答界面
  POST /ask       -> {"question": "..."} 返回 {"question":..,"answer":..}
"""
from flask import Flask, request, jsonify, render_template_string

import config
from rag import RAGService

app = Flask(__name__)
service = RAGService([config.PDF1, config.PDF2])

INDEX_PAGE = """
<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>金融问答系统</title></head>
<body>
<h2>金融招股说明书问答系统</h2>
<form id="f">
  <input name="q" size="60" placeholder="请输入问题">
  <button type="submit">提问</button>
</form>
<pre id="a"></pre>
<script>
document.getElementById('f').onsubmit = async function(e){
  e.preventDefault();
  const q = document.querySelector('input[name=q]').value;
  const r = await fetch('/ask', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({question:q})});
  const d = await r.json();
  document.getElementById('a').textContent = d.answer;
};
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_PAGE


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "question is empty"}), 400
    answer = service.answer(question)
    return jsonify({"question": question, "answer": answer})


if __name__ == "__main__":
    service.build()
    print(f"[deploy] 服务启动于 {config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT)
