# ============================================================
# 工单编号：人工智能NLP-RAG-Embedding模型微调任务
# 文件：build_db_ccf_finetuned.py
# 说明：用微调后的嵌入模型重建 CCF 向量库
# ============================================================
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

BASE = "/root/autodl-tmp/projects/RAG"
CCF_TXT_DIR = f"{BASE}/1_trans_data/ccf"
DB_PATH = f"{BASE}/ccf_vector_db_finetuned"
FINETUNED_MODEL = f"{BASE}/finetuned_bge"

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print(f"已清空旧向量库：{DB_PATH}")

# 加载 9 个年报
all_docs = []
files = sorted(os.listdir(CCF_TXT_DIR))
COMPANY_MAP = {
    "ccf_01": "平安银行2019年报",
    "ccf_02": "中国平安2019年报",
    "ccf_03": "招商银行2019年报",
    "ccf_04": "邮储银行2019年报",
    "ccf_05": "中信证券2020年报",
    "ccf_06": "中国人寿2020年报",
    "ccf_07": "中国太保2021年报",
    "ccf_08": "银河证券2021年报",
    "ccf_09": "国泰君安2021年报",
}

for fname in files:
    prefix = fname.replace(".txt", "")
    company = COMPANY_MAP.get(prefix, prefix)
    path = os.path.join(CCF_TXT_DIR, fname)
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = company
    all_docs.extend(docs)
    print(f"加载 {company}：{len(docs)} 个文档对象")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "；", "，", ""],
    length_function=len,
)
chunks = splitter.split_documents(all_docs)
print(f"\n共切分 {len(chunks)} 个文本块")

# 用微调后的模型
embeddings = HuggingFaceEmbeddings(
    model_name=FINETUNED_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH,
)
print(f"\n微调版 CCF 向量库构建完成：{DB_PATH}")
