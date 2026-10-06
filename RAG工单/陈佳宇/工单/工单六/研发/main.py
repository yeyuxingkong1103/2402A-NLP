from flask import Flask, render_template_string, request, jsonify
import pymupdf
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, DataType, utility, CollectionSchema
from rank_bm25 import BM25Okapi
import numpy as np

app = Flask(__name__)

# ========== 轻量化配置 ==========
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION_NAME = "rag_demo"
chunks_global = []
bm25 = None

# 连接Milvus
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)

# 加载PDF切分
def load_pdf_split(pdf_path):
    doc = pymupdf.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        paras = text.split("\n")
        for p in paras:
            p = p.strip()
            if len(p) > 20:
                chunks.append(p)
    return chunks

# 初始化向量库 + BM25
def init_knowledge_base():
    global chunks_global, bm25
    pdf_file = "招股说明书.pdf"
    chunks = load_pdf_split(pdf_file)
    chunks_global = chunks

    # BM25初始化
    tokenized_corpus = [doc.split(" ") for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    dim = 384
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)
    ]
    schema = CollectionSchema(fields=fields, description="混合检索知识库")
    coll = Collection(COLLECTION_NAME, schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    coll.create_index(field_name="vector", index_params=index_params)

    embeds = embed_model.encode(chunks)
    data = [
        embeds.tolist(),
        chunks
    ]
    coll.insert(data)
    coll.load()
    print("✅ 向量库 + BM25 初始化完成！")

# RRF倒数排名融合：融合【结果列表】，不再用Milvus主键
def rrf_fuse(vector_hit_list, bm25_idx_list, k=60):
    rank_scores = {}
    # 向量检索结果：key=文本内容，value=RRF分数
    for rank, hit in enumerate(vector_hit_list):
        txt = hit.entity.get("text")
        score = 1.0 / (k + rank + 1)
        rank_scores[txt] = rank_scores.get(txt,0) + score
    # BM25结果
    for rank, idx in enumerate(bm25_idx_list):
        txt = chunks_global[idx]
        score = 1.0 / (k + rank +1)
        rank_scores[txt] = rank_scores.get(txt,0) + score
    # 按分数降序
    sorted_items = sorted(rank_scores.items(), key=lambda x:x[1], reverse=True)
    return [item[0] for item in sorted_items]

# 检索入口
def search_func(query, search_type="hybrid", top_k=3):
    query_emb = embed_model.encode(query)
    tokenized_query = query.split(" ")
    coll = Collection(COLLECTION_NAME)

    vector_res = coll.search(
        data=[query_emb.tolist()],
        anns_field="vector",
        param={"metric_type":"L2", "params":{"nprobe":10}},
        limit=top_k,
        output_fields=["text"]
    )
    vector_hits = vector_res[0]
    vector_texts = [h.entity.get("text") for h in vector_hits]

    # BM25检索，返回数组下标
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_idx = np.argsort(bm25_scores)[::-1][:top_k].tolist()
    bm25_texts = [chunks_global[i] for i in bm25_idx]

    if search_type == "vector":
        final = vector_texts
    elif search_type == "bm25":
        final = bm25_texts
    else:
        # 混合检索RRF融合
        final = rrf_fuse(vector_hits, bm25_idx)[:top_k]

    # 兜底防护：空结果提示
    if not final:
        return ["未检索到相关文档片段"]
    return final

HTML_TPL = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>工单6 轻量化混合检索RAG</title>
<style>
.box{width:80%;margin:30px auto;}
.ipt{width:100%;height:100px;padding:10px;font-size:16px;}
.select{padding:8px;font-size:16px;margin-bottom:10px;}
.btn{padding:10px 20px;background:#2563eb;color:white;border:none;border-radius:4px;cursor:pointer;}
.res{margin-top:20px;padding:15px;background:#f5f5f5;white-space:pre-wrap;}
</style>
</head>
<body>
<div class="box">
    <h2>招股说明书问答｜混合检索系统</h2>
    <select class="select" id="search_type">
        <option value="vector">向量检索</option>
        <option value="bm25">BM25全文检索</option>
        <option value="hybrid" selected>混合检索(RRF融合)</option>
    </select>
    <textarea class="ipt" id="query" placeholder="请输入你的问题"></textarea>
    <br><br>
    <button class="btn" onclick="sendQuery()">提交检索</button>
    <div class="res" id="result"></div>
</div>
<script>
async function sendQuery(){
    const q = document.getElementById("query").value;
    const st = document.getElementById("search_type").value;
    const resp = await fetch("/api/query",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({query:q, search_type:st})
    });
    const data = await resp.json();
    document.getElementById("result").innerText = data.answer;
}
</script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(HTML_TPL)

@app.route("/api/query", methods=["POST"])
def api_query():
    req_data = request.get_json()
    q = req_data["query"]
    st = req_data["search_type"]
    res_texts = search_func(q, search_type=st, top_k=3)
    answer = "检索到的上下文：\n\n" + "\n\n=====\n\n".join(res_texts)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    init_knowledge_base()
    app.run(host="0.0.0.0", port=7860, debug=False)
