from flask import Flask, render_template_string, request
import pymupdf
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import requests

app = Flask(__name__)

# ===================== 配置区 =====================
PDF_PATH = "招股说明书.pdf"
COLLECTION_NAME = "rag_book"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# DeepSeek API配置，替换成你自己的key
DEEPSEEK_API_KEY = "sk-6e04da2625084194a695de2b19c13ad2"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
# ==================================================

# 加载嵌入模型
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# 使用MilvusClient
client = MilvusClient(uri=MILVUS_URI)

def to_python_float_list(vec):
    """把numpy向量转为原生Python float列表，解决Milvus类型报错"""
    return [float(x) for x in vec]

def load_pdf_and_split(pdf_path):
    """读取PDF，文本分块"""
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    full_text = re.sub(r"\s+", " ", full_text).strip()

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def create_milvus_collection():
    """创建集合，存在则先删除"""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=384)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    return client

def insert_chunks_to_milvus(chunks):
    vectors = embed_model.encode(chunks)
    data = [
        {"text": chunk, "vector": to_python_float_list(vec)}
        for chunk, vec in zip(chunks, vectors)
    ]
    client.insert(collection_name=COLLECTION_NAME, data=data)

# 初始化
print("正在读取PDF并分块...")
text_chunks = load_pdf_and_split(PDF_PATH)
print(f"一共拆分出 {len(text_chunks)} 个文本块")

print("正在创建Milvus集合并写入向量...")
create_milvus_collection()
insert_chunks_to_milvus(text_chunks)
print("✅ PDF向量入库完成！")

def get_llm_answer(context, question):
    prompt = f"""你是招股说明书问答助手，严格只使用下面参考内容回答问题，不要编造信息。
参考内容：
{context}

用户问题：{question}
直接简洁给出答案。
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
    res_json = resp.json()
    return res_json["choices"][0]["message"]["content"]


def search_rag(question):
    query_vec_np = embed_model.encode([question])[0]
    query_vec = to_python_float_list(query_vec_np)
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        anns_field="vector",
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=TOP_K,
        output_fields=["text"]
    )
    context = "\n".join([hit["entity"]["text"] for hit in res[0]])
    llm_ans = get_llm_answer(context, question)
    return llm_ans, context

# 网页前端
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>招股说明书RAG问答系统</title>
    <style>
        body {width:800px;margin:30px auto;font-family:system-ui;}
        textarea {width:100%;height:80px;padding:10px;font-size:16px;}
        button {margin-top:10px;padding:8px 20px;font-size:16px;cursor:pointer;}
        .answer {margin-top:20px;padding:15px;background:#f5f7fa;border-radius:8px;white-space:pre-wrap;}
        .llm-ans {background:#e8f4ff;}
    </style>
</head>
<body>
    <h2>📄 招股说明书RAG问答</h2>
    <form method="post">
        <textarea name="question" placeholder="请输入你的问题，例如：公司2024年归母净利润是多少？">{{question}}</textarea>
        <br>
        <button type="submit">提交问答</button>
    </form>
    {% if llm_answer %}
    <div class="answer llm-ans">
        <b>🤖 大模型回答：</b>
        <p>{{llm_answer}}</p>
    </div>
    <div class="answer">
        <b>📑 检索到的原文片段：</b>
        <p>{{context}}</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=["GET","POST"])
def index():
    question = ""
    llm_answer = ""
    context = ""
    if request.method == "POST":
        question = request.form.get("question","")
        llm_answer, context = search_rag(question)
    return render_template_string(HTML_TEMPLATE, question=question, llm_answer=llm_answer, context=context)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
