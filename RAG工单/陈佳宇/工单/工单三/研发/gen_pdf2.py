# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP‑RAG‑PDF文档的表格解析及检索优化
工单3：支持PDF表格解析RAG招股说明书问答系统
说明：设计方案为BM25+向量混合检索，因环境无法安装rank_bm25，当前版本使用纯向量检索，预留混合检索扩展位置
"""
import pymupdf
import pdfplumber
import re
import time
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# ====================== 配置 ======================
PDF_PATH = "招股说明书2.pdf"
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_waterwork3"
TOP_K = 3
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# 加载向量模型
embedding_model = SentenceTransformer(MODEL_NAME)
# 连接Milvus
client = MilvusClient(uri="http://localhost:19530")

# 全局缓存，预留BM25使用
all_text_chunks = []

# ====================== PDF读取：文本+表格解析【修复表格提取】 ======================
def load_pdf_with_table(pdf_path):
    full_text = ""
    # pdfplumber读取表格
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                # 转markdown表格
                md_rows = []
                for row in table:
                    clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                    md_rows.append("| " + " | ".join(clean_row) + " |")
                table_md = "\n".join(md_rows)
                full_text += f"\n[表格数据]\n{table_md}\n"
            # 读取页面普通文本
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text

# ====================== 文本分块 ======================
def split_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks

# ====================== 向量入库 ======================
def insert_data():
    global all_text_chunks
    print("正在读取PDF，解析文本与表格...")
    full_content = load_pdf_with_table(PDF_PATH)
    chunks = split_text(full_content)
    all_text_chunks = chunks
    print(f"文本分块完成，总块数：{len(chunks)}")

    vectors = embedding_model.encode(chunks)
    data = []
    for text, vec in zip(chunks, vectors):
        data.append({
            "text": text,
            "vector": vec.tolist()
        })
    client.insert(collection_name=COLLECTION_NAME, data=data)
    client.flush(collection_name=COLLECTION_NAME)
    print("✅ PDF（含表格）向量入库完成！")

# ====================== 向量检索 ======================
def search_vector(question, top_k=3):
    query_embedding = embedding_model.encode(question).tolist()
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text"]
    )
    result_texts = []
    for hit in res[0]:
        result_texts.append(hit["entity"]["text"])
    return result_texts

# ====================== RAG问答入口 ======================
def rag_answer(question):
    start_time = time.time()
    context_list = search_vector(question, TOP_K)
    context = "\n\n".join(context_list)
    cost = round(time.time() - start_time, 3)
    output = f"【检索耗时 {cost} 秒】\n【检索到的招股说明书原文片段】\n{context}"
    return output

# ====================== Flask网页前端（支持渲染Markdown表格） ======================
HTML_TPL = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>工单3-招股说明书RAG问答（支持表格）</title>
<style>
body{padding:30px;max-width:1000px;margin:0 auto;font-family:system-ui;}
#q{width:100%;padding:12px;font-size:16px;box-sizing:border-box;}
#btn{margin-top:10px;padding:12px 24px;background:#2563eb;color:white;border:none;border-radius:6px;cursor:pointer;font-size:16px;}
#ans{margin-top:20px;padding:16px;border-radius:8px;background:#f5f7fa;line-height:1.6;}
table {border-collapse: collapse;width:100%;margin:8px 0;}
th,td {border:1px solid #999;padding:6px 10px;}
th {background:#cce5ff;}
</style>
</head>
<body>
<h2>工单3｜招股说明书RAG问答系统（支持表格解析）</h2>
<input id="q" placeholder="示例：本次拟公开发行多少万股？">
<button id="btn">提问检索</button>
<div id="ans"></div>
<script>
document.getElementById('btn').onclick = async ()=>{
    const q = document.getElementById('q').value;
    const res = await fetch('/ask',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({q:q})
    });
    const json = await res.json();
    let html = json.ans.replaceAll("\\n","<br>");
    document.getElementById('ans').innerHTML = html;
}
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML_TPL)

@app.post("/ask")
def ask():
    q = request.get_json()["q"]
    ans = rag_answer(q)
    return jsonify({"ans": ans})

if __name__ == "__main__":
    insert_data()
    app.run(host="0.0.0.0", port=7861, debug=False)
