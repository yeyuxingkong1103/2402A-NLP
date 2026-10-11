# ============================================================
# 工单编号：人工智能NLP-RAG-RAG性能瓶颈识别与优化
# 文件：profile_rag_final2.py
# 说明：max_new_tokens=64 + 1块上下文 + 强精简 prompt
# ============================================================
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

MODEL_1_5B = "/root/autodl-tmp/models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
EMBED_PATH = "/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"
CCF_DB = "/root/autodl-tmp/projects/RAG/ccf_vector_db"

print("加载 1.5B (fp16, 不用量化更快)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_1_5B, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_1_5B,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
).eval()

print("加载嵌入模型 ...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_PATH,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma(persist_directory=CCF_DB, embedding_function=embedding_model)

def generate(prompt, max_new_tokens=64):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

QUESTIONS = [
    "平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？",
    "招商银行2019年年报中提到的核心竞争力是什么？",
    "中信证券2020年年报中的主要业务收入构成是什么？",
    "中国人寿2020年年报中的投资策略是什么？",
    "中国太保2021年年报中的保险业务发展重点是什么？",
]

def ask(q):
    stages = {}
    t0 = time.time()
    # 只取 1 块
    docs = vectordb.similarity_search(q, k=1)
    stages["retrieval"] = time.time() - t0

    t0 = time.time()
    # 上下文截断 250 字 + 极简 prompt
    context = docs[0].page_content[:250] if docs else ""
    prompt = f"根据以下内容简答（不超过50字）：\n{context}\n问：{q}\n答："
    answer = generate(prompt, max_new_tokens=64)
    stages["generation"] = time.time() - t0

    stages["total"] = sum(stages.values())
    return stages, answer[:80]

print("=" * 70)
print("最终极速版（1.5B fp16 + 64 tokens + 1 块上下文）")
print("=" * 70)
print(f"{'问题':<30} {'检索':>8} {'生成':>8} {'总计':>8}")
print("-" * 70)

all_stages = []
for q in QUESTIONS:
    stages, ans = ask(q)
    all_stages.append(stages)
    print(f"{q[:28]:<30} {stages['retrieval']:>7.2f}s {stages['generation']:>7.2f}s {stages['total']:>7.2f}s")
    print(f"    回答：{ans}")

print("-" * 70)
avg = {k: sum(s[k] for s in all_stages) / len(all_stages) for k in all_stages[0]}
print(f"{'平均':<30} {avg['retrieval']:>7.2f}s {avg['generation']:>7.2f}s {avg['total']:>7.2f}s")
print("=" * 70)
print(f"\n优化前（7B）: 10.83s")
print(f"最终版: {avg['total']:.2f}s")
print(f"是否达标（≤3s）：{'✅ 达标' if avg['total'] <= 3.0 else '❌ 未达标'}")
