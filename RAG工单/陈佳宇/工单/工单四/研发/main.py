# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP‑RAG‑图像内容解析及检索优化
工单4：PDF图像多模态RAG问答系统
"""
import os
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymilvus import MilvusClient
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

COLLECTION_NAME = "rag_waterwork4"
IMG_FOLDER = "pdf_images"
TOP_K = 2
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

print("正在加载CLIP多模态模型...")
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
client = MilvusClient(uri="http://localhost:19530")

# 转为一维纯Python float列表
def to_python_float_list(tensor):
    arr = tensor.squeeze().detach().cpu().numpy()
    raw_list = arr.tolist()
    return [float(x) for x in raw_list]

def image_embedding(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt", padding=True)
    out = model.get_image_features(**inputs)
    img_feat = out.pooler_output
    vec = to_python_float_list(img_feat)
    return vec

def insert_image_data():
    print("开始读取图片并向量化入库...")
    img_files = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(".png")]
    insert_data = []
    for fname in img_files:
        full_path = os.path.join(IMG_FOLDER, fname)
        pil_img = Image.open(full_path).convert("RGB")
        vec = image_embedding(pil_img)
        insert_data.append({
            "image_name": fname,
            "image_path": full_path,
            "vector": vec
        })
    if len(insert_data) > 0:
        client.insert(collection_name=COLLECTION_NAME, data=insert_data)
        client.flush(collection_name=COLLECTION_NAME)
    print(f"✅ {len(insert_data)} 张图片向量入库完成！")

def text_embedding(text):
    # 增加 truncation=True，自动截断超长文本，适配CLIP最大77token限制
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    out = model.get_text_features(**inputs)
    text_feat = out.pooler_output
    vec = to_python_float_list(text_feat)
    return vec

def multimodal_search(question, top_k=2):
    q_vec = text_embedding(question)
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=top_k,
        output_fields=["image_name", "image_path"]
    )
    out = []
    for hit in res[0]:
        out.append({
            "score": round(hit["distance"],4),
            "image_name": hit["entity"]["image_name"],
            "image_path": hit["entity"]["image_path"]
        })
    return out

def rag_query(question):
    t0 = time.time()
    result_list = multimodal_search(question, TOP_K)
    cost = round(time.time()-t0,3)
    resp_text = f"【检索耗时：{cost}秒】\n"
    for idx,item in enumerate(result_list):
        resp_text += f"\n---匹配图片{idx+1}---\n图片文件名：{item['image_name']}\n相似度得分：{item['score']}\n"
    return resp_text

HTML_TPL = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>工单4｜PDF图像多模态RAG检索</title>
<style>
body{padding:30px;max-width:1000px;margin:0 auto;}
#q{width:100%;padding:12px;font-size:16px;box-sizing:border-box;}
#btn{margin-top:10px;padding:12px 22px;background:#1f6ed9;color:white;border:none;border-radius:6px;font-size:15px;cursor:pointer;}
#ans{margin-top:20px;background:#f4f7fb;padding:18px;border-radius:8px;white-space:pre-wrap;line-height:1.7;}
</style>
</head>
<body>
<h2>工单4｜PDF图像内容解析多模态RAG检索系统</h2>
<p>示例：武汉力源招股说明书里销售部有几个部门？</p>
<input id="q" placeholder="输入针对PDF图片的问题">
<button id="btn">提问检索</button>
<div id="ans"></div>
<script>
document.getElementById('btn').onclick = async ()=>{
    const q = document.getElementById("q").value;
    const res = await fetch("/ask",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({"q":q})
    });
    const json = await res.json();
    document.getElementById("ans").textContent = json.ans;
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
    ans = rag_query(q)
    return jsonify({"ans":ans})

if __name__ == "__main__":
    insert_image_data()
    app.run(host="0.0.0.0",port=7862,debug=False)
