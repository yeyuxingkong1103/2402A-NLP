# ============================================================
# 工单编号：人工智能NLP-RAG-LightRAG 优化任务
# 文件：build_lightrag_index.py
# 说明：用 LightRAG 对招股说明书关键章节构建知识图谱索引
#       路径A：仅处理前 2 万字符，10-15 分钟完成
# ============================================================
import os
import asyncio
import torch
import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------- 配置 ----------------
WORKING_DIR = "/root/autodl-tmp/projects/RAG/lightrag_index"
TXT1 = "/root/autodl-tmp/projects/RAG/1_trans_data/招股说明书1.txt"
TXT2 = "/root/autodl-tmp/projects/RAG/1_trans_data/招股说明书2.txt"
MODEL_PATH = "/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
EMBED_PATH = "/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"
MAX_CHARS = 20000   # ★ 简化：每个 PDF 只取前 2 万字符

os.makedirs(WORKING_DIR, exist_ok=True)

# ---------------- 加载 Qwen ----------------
print("加载 Qwen2.5-7B ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
).eval()

def qwen_generate(prompt, max_new_tokens=1024):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

# ---------------- ★ 强化格式约束的 LLM 调用 ----------------
FORMAT_HINT = """
【重要】严格按照以下格式输出，不要用 Markdown 表格，不要用分隔线。

实体输出格式（每行一个）：
entity<|>实体名<|>实体类型<|>实体描述

关系输出格式（每行一个）：
relation<|>源实体<|>目标实体<|>关系描述<|>关键词

必须使用 <|> 作为分隔符。不使用表格符号 | 或 --------。
"""

async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    full_prompt = FORMAT_HINT + "\n\n" + prompt
    return qwen_generate(full_prompt, max_new_tokens=1024)

# ---------------- 加载嵌入模型 ----------------
print("加载嵌入模型 ...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_PATH,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

async def embedding_func(texts: list) -> np.ndarray:
    return np.array(embedding_model.embed_documents(texts))

# ---------------- 初始化 LightRAG ----------------
print("初始化 LightRAG ...")
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=512,
        max_token_size=512,
        func=embedding_func,
    ),
)

# ---------------- 读取并截取文本 ----------------
print(f"读取招股书1（前 {MAX_CHARS} 字符）...")
with open(TXT1, "r", encoding="utf-8") as f:
    text1 = f.read()[:MAX_CHARS]

print(f"读取招股书2（前 {MAX_CHARS} 字符）...")
with open(TXT2, "r", encoding="utf-8") as f:
    text2 = f.read()[:MAX_CHARS]

# ---------------- 插入文档构建索引 ----------------
async def main():
    from lightrag.kg.shared_storage import initialize_pipeline_status
    await rag.initialize_storages()
    await initialize_pipeline_status()

    print("\n开始插入招股书1 ...")
    await rag.ainsert(text1)
    print("招股书1 完成")

    print("\n开始插入招股书2 ...")
    await rag.ainsert(text2)
    print("招股书2 完成")

    print(f"\n✅ LightRAG 索引构建完成！保存在：{WORKING_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
