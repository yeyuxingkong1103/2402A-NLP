# ============================================================
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 项目名称：PDF文档的功能测试及评估
# 文件：build_db_ccf.py
# 说明：为 ccf_competition 的 9 个年报构建独立向量库
# ============================================================
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
from collections import Counter

BASE = "/root/autodl-tmp/projects/RAG"
CCF_TXT_DIR = f"{BASE}/1_trans_data/ccf"
DB_PATH = f"{BASE}/ccf_vector_db"

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print(f"已清空旧 CCF 向量库：{DB_PATH}")

# 加载 9 个年报文本
all_docs = []
files = sorted(os.listdir(CCF_TXT_DIR))
# 文件编号到公司名的映射
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

# 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "；", "，", ""],
    length_function=len,
)

chunks = splitter.split_documents(all_docs)
print(f"\n共切分 {len(chunks)} 个文本块")

src_count = Counter(c.metadata.get("source", "未知") for c in chunks)
for k, v in sorted(src_count.items()):
    print(f"  {k}：{v} 块")

# 向量化
embeddings = HuggingFaceEmbeddings(
    model_name="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH,
)
print(f"\nCCF 向量库构建完成：{DB_PATH}")