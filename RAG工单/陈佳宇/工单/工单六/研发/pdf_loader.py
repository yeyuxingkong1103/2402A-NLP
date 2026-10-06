from pymilvus import connections, Collection
import pymupdf
import numpy as np

COLLECTION_NAME = "rag_workorder5"
DIM = 128

# 简易文本向量化（不需要任何预训练模型）
def simple_embedding(text: str):
    np.random.seed(hash(text) % 2147483647)
    vec = np.random.randn(DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()

# 读取PDF
def read_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 文本分块
def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

print("正在读取招股说明书PDF...")
pdf_text = read_pdf("招股说明书.pdf")
text_chunks = split_text(pdf_text)
print(f"切分得到 {len(text_chunks)} 个文本块")

# 生成向量
print("正在生成向量...")
vectors = []
for chunk in text_chunks:
    vectors.append(simple_embedding(chunk))

# 连接Milvus
connections.connect("default", host="localhost", port="19530")
coll = Collection(COLLECTION_NAME)

# 插入顺序：主键auto_id不用传，只需要 vector, text
insert_data = [
    vectors,
    text_chunks
]

print("正在写入Milvus向量库...")
coll.insert(insert_data)
coll.flush()

print("✅ PDF向量入库完成！")
