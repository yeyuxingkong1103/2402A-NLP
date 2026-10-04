# 工单2 main.py
# 工单2：RAG招股说明书问答系统
import pymupdf
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from flask import Flask, request, render_template_string, jsonify
import re

app = Flask(__name__)

# ====================== 配置 ======================
PDF_PATH = "招股说明书.pdf"
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_waterwork2"
TOP_K = 3
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# 加载向量模型
embedding_model = SentenceTransformer(MODEL_NAME)
# 连接Milvus
client = MilvusClient(uri="http://localhost:19530")

# ====================== PDF读取与分块 ======================
def load_pdf_and_split(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        page_text = page.get_text()
        full_text += page_text
    # 简单清洗
    full_text = re.sub(r"\s+", " ", full_text)
    # 文本分块
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunk = full_text[start:end]
        chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks

# ====================== 向量入库 ======================
def insert_data():
    print("正在读取PDF并清洗文本...")
    chunks = load_pdf_and_split(PDF_PATH)
    print(f"文本分块完成，总块数：{len(chunks)}")
    # 向量化
    vectors = embedding_model.encode(chunks)
    # 组装Milvus数据
    data = []
    for text, vec in zip(chunks, vectors):
        data.append({
            "text": text,
            "vector": vec.tolist()
        })
    # 插入
    client.insert(collection_name=COLLECTION_NAME, data=data)
    client.flush(collection_name=COLLECTION_NAME)
    print("✅ PDF向量入库完成！")

# 执行入库
insert_data()

# ====================== 检索函数 ======================
def search_rag(question):
    q_vec = embedding_model.encode(question).tolist()
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=TOP_K,
        output_fields=["text"]
    )
    # 拼接检索结果
    context = ""
    for hit in res[0]:
        context += hit["entity"]["text"] + "\n\n"
    answer = f"【检索到的招股说明书原文片段】\n{context}"
    return answer

# ====================== Flask网页 ======================
HTML_TPL = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>招股说明书RAG问答</title>
<style>
body{padding:30px;max-width:1000px;margin:0 auto;font-family:system-ui;}
#q{width:100%;padding:12px;font-size:16px;}
#btn{margin-top:10px;padding:12px 24px;background:#2563eb;color:white;border:none;border-radius:6px;cursor:pointer;}
#ans{margin-top:20px;white-space:pre-wrap;line-height:1.6;background:#f5f7fa;padding:16px;border-radius:8px;}
</style>
</head>
<body>
<h2>招股说明书 RAG 问答系统</h2>
<input id="q" placeholder="请输入你的问题，例如：公司主营业务是什么？">
<button id="btn">提问</button>
<div id="ans"></div>
<script>
document.getElementById('btn').onclick = async ()=>{
    const q = document.getElementById('q').value;
    const res = await fetch('/ask',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({q})
    });
    const json = await res.json();
    document.getElementById('ans').innerText = json.ans;
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
    ans = search_rag(q)
    return jsonify({"ans": ans})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)

工单 2/
├── 设计 /
├── 研发 /
│   ├── create_collection.py   # Milvus 集合创建脚本
│   ├── main.py                # 项目主程序 + Flask 网页
│   └── 招股说明书.pdf         # 模拟招股说明书 PDF
├── 测试 /
├── 优化 /
└── 部署 /


## 环境安装
```bash
pip install flask pymupdf transformers torch sentence-transformers pymilvus

## 启动步骤

1. 启动 Milvus 服务
2. 创建向量集合：`python create_collection.py`
3. 启动 RAG 问答系统：`python main.py`
4. 浏览器访问：[http://127.0.0.1:7860](http://127.0.0.1:7860)

## 功能说明

输入问题，系统会在招股说明书 PDF 中检索相关原文片段并返回。
示例问题：

1. 公司主营业务是什么？
2. 本次募集资金用途？
3. 公司最近三年营收情况？


## 1. 设计阶段 /设计/design.md
```markdown
# 项目设计文档
## 1. 项目目标
构建基于RAG的招股说明书PDF问答系统，用户在网页输入问题，系统检索PDF文档内相关内容并返回原文片段。

## 2. 整体架构
整体分为四层：文档层 → 文本处理层 → 向量存储检索层 → Web交互层
1. 文档层：模拟招股说明书PDF文件
2. 文本处理层：PDF读取、文本清洗、文本分块
3. 向量层：sentence-transformers生成向量，Milvus向量库存储、相似度检索
4. Web层：Flask后端 + HTML前端，浏览器交互问答

## 3. 核心参数设计
- 向量模型：all-MiniLM-L6-v2，向量维度384
- 文本分块：块大小512，重叠50
- 向量数据库：Milvus v3.0，索引类型IVF_FLAT，距离度量L2
- 检索返回TopK=3条相关文本

## 4. 接口设计
- GET `/`：返回问答网页首页
- POST `/ask`：接收用户问题，调用RAG检索，返回检索到的原文

## 5. 数据库Schema设计
集合名称：rag_waterwork2
|字段|类型|说明|
| ---- | ---- | ---- |
|id|INT64|主键，自动生成|
|vector|FLOAT_VECTOR(384)|文本向量|
|text|VARCHAR|PDF原文片段|
