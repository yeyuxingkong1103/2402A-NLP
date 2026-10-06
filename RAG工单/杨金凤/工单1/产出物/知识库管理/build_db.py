from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载解析后的文本
loader = TextLoader(
    "/root/autodl-tmp/projects/RAG/1_trans_data/招股说明书1.txt",
    encoding="utf-8"
)
docs = loader.load()

# 2. 中文友好的递归分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", "，", ""],
    length_function=len,
)
chunks = splitter.split_documents(docs)
print(f"共切分 {len(chunks)} 个文本块")

# 3. 嵌入并持久化到 Chroma
embeddings = HuggingFaceEmbeddings(
    model_name="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="/root/autodl-tmp/projects/RAG/2_vector_db",
)
vectordb.persist()
print("向量库构建完成")