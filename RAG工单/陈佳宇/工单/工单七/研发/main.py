from flask import Flask, render_template_string, request, jsonify
import pymupdf
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, DataType, utility, CollectionSchema
from rank_bm25 import BM25Okapi
import numpy as np

app = Flask(__name__)

# ========== 配置 ==========
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION_NAME = "rag_eval_demo"
chunks_global = []
bm25 = None

# Milvus连接
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)

def load_pdf_split(pdf_path):
    """加载PDF，文本切分"""
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

def init_knowledge_base(pdf_file):
    global chunks_global, bm25
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
    schema = CollectionSchema(fields=fields, description="工单7‑RAG评估知识库")
    coll = Collection(COLLECTION_NAME, schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    coll.create_index(field_name="vector", index_params=index_params)

    embeds = embed_model.encode(chunks)
    coll.insert([embeds.tolist(), chunks])
    coll.load()
    print("✅知识库初始化完成，文档块数量：", len(chunks_global))

def rrf_fuse(vector_hit_list, bm25_idx_list, k=60):
    rank_scores = {}
    for rank, hit in enumerate(vector_hit_list):
        txt = hit.entity.get("text")
        score = 1.0 / (k + rank + 1)
        rank_scores[txt] = rank_scores.get(txt, 0) + score
    for rank, idx in enumerate(bm25_idx_list):
        txt = chunks_global[idx]
        score = 1.0 / (k + rank + 1)
        rank_scores[txt] = rank_scores.get(txt, 0) + score
    sorted_items = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items]

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

    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_idx = np.argsort(bm25_scores)[::-1][:top_k].tolist()
    bm25_texts = [chunks_global[i] for i in bm25_idx]

    if search_type == "vector":
        final = vector_texts
    elif search_type == "bm25":
        final = bm25_texts
    else:
        final = rrf_fuse(vector_hits, bm25_idx)[:top_k]

    if not final:
        return ["未检索到相关文档片段"]
    return final

# ====================== 工单7评估模块：批量测试10条问题 ======================
def batch_evaluate(question_list, search_type="hybrid", top_k=3):
    """
    question_list: [(问题, 参考相关文本集合), ... ]
    返回每条：问题，检索结果，召回的相关数量，总相关数，召回率，精确率
    """
    report = []
    for q, ground_truth_rel in question_list:
        retrieved = search_func(q, search_type=search_type, top_k=top_k)
        hit_count = 0
        total_rel = len(ground_truth_rel)
        for gt in ground_truth_rel:
            for ret_txt in retrieved:
                if gt in ret_txt:
                    hit_count +=1
                    break
        recall = hit_count / total_rel if total_rel >0 else 0.0
        precision = hit_count / len(retrieved) if len(retrieved)>0 else 0.0
        report.append({
            "question": q,
            "retrieved": retrieved,
            "hit": hit_count,
            "total_gt": total_rel,
            "recall": round(recall,4),
            "precision": round(precision,4)
        })
    return report

# ========== Web前端页面（修复JS符号bug） ==========
HTML_TPL = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>工单7-RAG测试评估系统</title>
<style>
.box{width:82%;margin:30px auto;}
.ipt{width:100%;height:110px;padding:10px;font-size:16px;}
.select{padding:8px;font-size:16px;margin-bottom:10px;}
.btn{padding:10px 22px;background:#2563eb;color:#fff;border:none;border-radius:4px;cursor:pointer;}
.res{margin-top:15px;padding:12px;background:#f3f4f6;white-space:pre-wrap;}
</style>
</head>
<body>
<div class="box">
    <h2>RAG检索测试｜工单7功能测试评估</h2>
    <select class="select" id="search_type">
        <option value="vector">向量检索</option>
        <option value="bm25">BM25全文检索</option>
        <option value="hybrid" selected>混合检索(RRF融合)</option>
    </select>
    <br>
    <textarea class="ipt" id="query" placeholder="输入测试问题"></textarea>
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
    req = request.get_json()
    q = req["query"]
    st = req["search_type"]
    res_texts = search_func(q, search_type=st, top_k=3)
    answer = "【检索返回片段】：\n\n" + "\n\n=====\n\n".join(res_texts)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    # PDF放在当前研发文件夹，修改此处为你的pdf文件名
    PDF_NAME = "招股说明书.pdf"
    init_knowledge_base(PDF_NAME)

    # ========= 示例：10条测试用例模板，工单7在这里填入ccf数据集的问题和参考真值 =========
    # 格式：(问题字符串, [相关参考文本片段列表])
    test_questions = [
        ("示例问题1", ["参考真值片段1"]),
        ("示例问题2", ["参考真值片段2"]),
        ("示例问题3", ["参考真值片段3"]),
        ("示例问题4", ["参考真值片段4"]),
        ("示例问题5", ["参考真值片段5"]),
        ("示例问题6", ["参考真值片段6"]),
        ("示例问题7", ["参考真值片段7"]),
        ("示例问题8", ["参考真值片段8"]),
        ("示例问题9", ["参考真值片段9"]),
        ("示例问题10", ["参考真值片段10"]),
    ]

    # 执行批量评估（混合检索）
    eval_result = batch_evaluate(test_questions, search_type="hybrid", top_k=3)
    print("\n==================== 工单7批量评估输出 ====================")
    for idx, item in enumerate(eval_result):
        print(f"\n【问题{idx+1}】{item['question']}")
        print(f"检索结果：{item['retrieved']}")
        print(f"命中相关片段：{item['hit']} / {item['total_gt']}")
        print(f"召回率={item['recall']}，精确率={item['precision']}")
    print("==========================================================\n")

    app.run(host="0.0.0.0", port=7860, debug=False)
