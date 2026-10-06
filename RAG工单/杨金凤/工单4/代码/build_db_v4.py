# ============================================================
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 项目名称：PDF文档的图像内容解析及检索优化
# 文件：build_db_v4.py
# 说明：重建向量库，包含两份 PDF 的文本、表格和图像描述
# ============================================================
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import shutil
import os


DB_PATH = "/root/autodl-tmp/projects/RAG/2_vector_db"
BASE = "/root/autodl-tmp/projects/RAG"

# 清空旧向量库
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print(f"已清空旧向量库：{DB_PATH}")

# 加载数据源：文本 + 图像描述
sources = [
    ("招股说明书1", f"{BASE}/1_trans_data/招股说明书1.txt"),
    ("招股说明书2", f"{BASE}/1_trans_data/招股说明书2.txt"),
    ("图像描述",   f"{BASE}/1_trans_data/图像描述.txt"),
]

all_docs = []
for name, path in sources:
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = name
    all_docs.extend(docs)
    print(f"加载 {name}：{len(docs)} 个文档对象")

# 分块：表格和图像描述作为原子单元不切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=[
        "### 图像描述结束 ###",   # 图像描述结束标记
        "### 表格",               # 表格开始标记
        "\n\n", "\n", "。", "；", "，", "",
    ],
    length_function=len,
    keep_separator=True,
)

chunks = splitter.split_documents(all_docs)
print(f"共切分 {len(chunks)} 个文本块")

from collections import Counter
src_count = Counter(c.metadata.get("source", "未知") for c in chunks)
for k, v in src_count.items():
    print(f"  {k}：{v} 块")

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
print("向量库构建完成")