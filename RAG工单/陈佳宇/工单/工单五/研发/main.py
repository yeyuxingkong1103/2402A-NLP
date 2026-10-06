from flask import Flask, request, render_template_string
from pymilvus import connections, Collection
import numpy as np
from openai import OpenAI

app = Flask(__name__)
COLLECTION_NAME = "rag_workorder5"
DIM = 128

# 简易向量函数（不需要任何模型）
def simple_embedding(text: str):
    np.random.seed(hash(text) % 2147483647)
    vec = np.random.randn(DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()

# 全局对话历史
chat_history = []

# DeepSeek 接口
client = OpenAI(
    api_key="sk-6e04da2625084194a695de2b19c13ad2",
    base_url="https://api.deepseek.com/v1"
)

# 查询改写
def query_rewrite(user_q: str, history):
    prompt = f"""把用户问题改写为完整独立问题，不要丢失上下文。
对话历史：{history}
用户问题：{user_q}
只输出改写后的问题，不要多余解释。"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1
    )
    return resp.choices[0].message.content

# RAG检索
def search_milvus(query):
    connections.connect("default", host="localhost", port="19530")
    coll = Collection(COLLECTION_NAME)
    coll.load()
    q_vec = simple_embedding(query)
    res = coll.search(
        data=[q_vec],
        anns_field="vector",
        param={"metric_type":"L2","params":{"nprobe":10}},
        limit=3,
        output_fields=["text"]
    )
    chunks = []
    for hit in res[0]:
        chunks.append(hit.entity.get("text"))
    return "\n".join(chunks)

# 问答接口
@app.route("/ask", methods=["POST"])
def ask():
    global chat_history
    user_input = request.form.get("user_input")
    rewrite_q = query_rewrite(user_input, chat_history)
    context = search_milvus(rewrite_q)
    prompt = f"""基于下面招股说明书参考内容回答用户问题。
参考内容：{context}
用户问题：{user_input}
如果参考内容没有相关信息，就说明文档没有相关内容。"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    answer = resp.choices[0].message.content
    chat_history.append({"user":user_input,"assistant":answer})
    return answer

# 网页首页
html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>招股说明书RAG问答</title>
</head>
<body>
    <h2>招股说明书问答系统</h2>
    <div>
        <textarea id="question" placeholder="请输入你的问题" rows="4" cols="60"></textarea>
        <br>
        <button onclick="sendQuery()">提交提问</button>
    </div>
    <div id="answer" style="margin-top:20px;white-space:pre-wrap;"></div>
<script>
async function sendQuery(){
    const q = document.getElementById("question").value;
    const res = await fetch("/ask",{
        method:"POST",
        body:new URLSearchParams({"user_input":q})
    });
    const ans = await res.text();
    document.getElementById("answer").innerText = ans;
}
</script>
</body>
</html>
"""
@app.route("/")
def index():
    return render_template_string(html_template)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
