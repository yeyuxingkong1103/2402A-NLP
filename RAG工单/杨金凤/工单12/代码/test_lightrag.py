# ============================================================
# 工单编号：人工智能NLP-RAG-LightRAG 优化任务
# 文件：test_lightrag.py
# 说明：用 LightRAG 检索工单指定的 16 个问题
# ============================================================
import os
import asyncio
import json
import torch
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

WORKING_DIR = "/root/autodl-tmp/projects/RAG/lightrag_index"
MODEL_PATH = "/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
EMBED_PATH = "/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"
OUTPUT_FILE = "/root/autodl-tmp/projects/RAG/lightrag_results.json"

QUESTIONS = [
    "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
    "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
    "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
    "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
    "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
    "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
    "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
    "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
]

print("加载 Qwen ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
).eval()

def qwen_generate(prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    return qwen_generate(prompt, max_new_tokens=512)

print("加载嵌入模型 ...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_PATH,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

async def embedding_func(texts: list) -> np.ndarray:
    return np.array(embedding_model.embed_documents(texts))

print("加载 LightRAG 索引 ...")
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(embedding_dim=512, max_token_size=512, func=embedding_func),
)

async def main():
    from lightrag.kg.shared_storage import initialize_pipeline_status
    await rag.initialize_storages()
    await initialize_pipeline_status()

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/16] {q[:50]}...")
        try:
            answer = await rag.aquery(q, param=QueryParam(mode="hybrid"))
            results.append({"id": i, "question": q, "answer": answer})
            print(f"  回答：{answer[:150]}...")
        except Exception as e:
            print(f"  错误：{e}")
            results.append({"id": i, "question": q, "answer": f"ERROR: {e}"})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 完成，结果保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
